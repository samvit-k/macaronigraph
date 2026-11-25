// Holds one tab MediaStream for the whole session; records audio and samples JPEG frames in memory only.
/* global cropPx */
const API = "http://localhost:8765";
const ENCODE_TIMEOUT_MS = 3000;
const video = document.getElementById("v");

let stream = null, audioCtx = null, analyser = null, lost = null;
let item = null;

const send = (msg) => chrome.runtime.sendMessage({ target: "background", ...msg }).catch(() => {});

async function startCapture({ streamId, viewport }) {
  if (stream) throw new Error("capture already running");
  const maxW = Math.min(2560, Math.round(viewport.w * viewport.dpr));
  const maxH = Math.min(2560, Math.round(viewport.h * viewport.dpr));
  // Consume the stream id immediately (it expires within seconds).
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } },
    video: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId,
                          maxWidth: maxW, maxHeight: maxH, maxFrameRate: 30 } },
  });
  lost = null;
  for (const tr of stream.getTracks()) {
    tr.onended = () => { lost = `${tr.kind} track ended`; send({ type: "capture-lost", detail: lost }); };
  }
  // Tab capture mutes the tab for the user; route its audio back to the speakers.
  audioCtx = new AudioContext();
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  src.connect(analyser);
  src.connect(audioCtx.destination);
  await audioCtx.resume();
  video.srcObject = stream;
  await video.play();
  return { ok: true, audioTracks: stream.getAudioTracks().length, videoTracks: stream.getVideoTracks().length,
           settings: stream.getVideoTracks()[0]?.getSettings() };
}

function streamLive() {
  return !!stream && stream.getTracks().length > 0 && stream.getTracks().every((t) => t.readyState === "live");
}

function sampleFrame(it) {
  const wall = Date.now();
  const fw = video.videoWidth, fh = video.videoHeight;
  if (!fw || !fh) { it.skipped.push("no_frame_size"); return; }
  const c = it.crop ? cropPx(fw, fh, it.crop.viewport, it.crop.rect) : { x: 0, y: 0, w: fw, h: fh };
  if (c.w < 8 || c.h < 8) { it.skipped.push(`crop_too_small:${c.w}x${c.h}`); return; }
  it.lastCrop = { ...c, frame: `${fw}x${fh}` };
  const canvas = new OffscreenCanvas(c.w, c.h); // one canvas per frame: no shared-canvas races
  canvas.getContext("2d").drawImage(video, c.x, c.y, c.w, c.h, 0, 0, c.w, c.h);
  const p = canvas.convertToBlob({ type: "image/jpeg", quality: 0.9 })
    .then((blob) => { it.frames.push({ wall, blob }); })
    .catch((e) => { it.failed.push(`${e.name}: ${e.message}`); });
  it.pending.add(p);
  p.finally(() => it.pending.delete(p));
}

function beginItem({ intervalMs, crop }) {
  if (!stream) throw new Error("no capture stream");
  if (!streamLive()) throw new Error(`capture stream not live (${lost || "unknown"})`);
  if (item) throw new Error("item already in progress");
  const audioTracks = stream.getAudioTracks();
  const it = { chunks: [], frames: [], pending: new Set(), failed: [], skipped: [], attempts: 0,
               peak: 0, intervalMs, crop, recorder: null, lastCrop: null };
  if (audioTracks.length) {
    it.recorder = new MediaRecorder(new MediaStream(audioTracks), { mimeType: "audio/webm;codecs=opus" });
    it.recorder.ondataavailable = (e) => e.data.size && it.chunks.push(e.data);
    it.recorder.start(1000);
  }
  it.recStartWall = Date.now();
  const buf = new Float32Array(analyser.fftSize);
  it.levelTimer = setInterval(() => {
    analyser.getFloatTimeDomainData(buf);
    for (const x of buf) if (Math.abs(x) > it.peak) it.peak = Math.abs(x);
  }, 100);
  const tick = () => { it.attempts++; sampleFrame(it); };
  tick();
  it.timer = setInterval(tick, intervalMs);
  item = it;
  return { ok: true, recStartWall: it.recStartWall };
}

async function endItem({ meta, upload }) {
  const it = item;
  if (!it) return { ok: false, error: "no item in progress" };
  item = null;
  clearInterval(it.timer); clearInterval(it.levelTimer);
  if (it.recorder && it.recorder.state !== "inactive") {
    await new Promise((r) => { it.recorder.onstop = r; it.recorder.stop(); });
  }

  const inflight = [...it.pending];
  let timedOut = false;
  await Promise.race([
    Promise.allSettled(inflight),
    new Promise((r) => setTimeout(() => { timedOut = true; r(); }, ENCODE_TIMEOUT_MS)),
  ]);
  const frames = it.frames.slice().sort((a, b) => a.wall - b.wall);
  const dropped = timedOut ? it.pending.size : 0;
  const walls = frames.map((f) => f.wall);
  const audioBlob = it.chunks.length ? new Blob(it.chunks, { type: "audio/webm" }) : null;
  const fullMeta = {
    ...meta,
    rec_start_wall: it.recStartWall,
    frame_walls: walls,
    frame_interval_ms: it.intervalMs,
    frame_interval_actual_ms: walls.length > 1 ? Math.round((walls.at(-1) - walls[0]) / (walls.length - 1)) : null,
    frames_attempted: it.attempts,
    frames_sampled: walls.length,
    frames_skipped: it.skipped.length,
    frames_encode_failed: it.failed.length,
    frames_dropped_timeout: dropped,
    frame_issues: [...new Set([...it.skipped, ...it.failed])].slice(0, 10),
    crop_px: it.lastCrop,
    audio_peak_db: it.peak > 0 ? Math.round(200 * Math.log10(it.peak)) / 10 : -200,
    audio_tracks: stream ? stream.getAudioTracks().length : 0,
    audio_bytes_client: audioBlob ? audioBlob.size : 0,
    stream_live: streamLive(),
    capture_lost: lost,
  };
  if (lost && fullMeta.playback === "ok") { fullMeta.playback = "capture_lost"; fullMeta.playback_detail = lost; }

  const fd = new FormData();
  fd.append("meta", JSON.stringify(fullMeta));
  if (upload && audioBlob) fd.append("audio", audioBlob, "audio.webm");
  if (upload) frames.forEach((f, i) => fd.append("frames", f.blob, `f${i}.jpg`));
  it.chunks.length = 0; it.frames.length = 0;
  const res = await fetch(`${API}/api/process`, { method: "POST", body: fd });
  if (!res.ok) return { ok: false, error: `backend ${res.status}: ${await res.text()}` };
  const { frame_walls, ...summary } = fullMeta; // eslint-disable-line no-unused-vars
  return { ok: true, result: await res.json(), meta: summary };
}

async function postMetaOnly({ meta }) {
  const fd = new FormData();
  fd.append("meta", JSON.stringify(meta));
  const res = await fetch(`${API}/api/process`, { method: "POST", body: fd });
  if (!res.ok) return { ok: false, error: `backend ${res.status}: ${await res.text()}` };
  return { ok: true, result: await res.json() };
}

function stopCapture() {
  if (item) { clearInterval(item.timer); clearInterval(item.levelTimer); item = null; }
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  audioCtx?.close();
  audioCtx = null;
  video.srcObject = null;
  return { ok: true };
}

const handlers = { "start-capture": startCapture, "begin-item": beginItem, "end-item": endItem,
                   "post-meta": postMetaOnly, "stop-capture": stopCapture,
                   "ping": () => ({ ok: true, live: streamLive(), lost }) };

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.target !== "offscreen") return;
  Promise.resolve()
    .then(() => handlers[msg.type](msg))
    .then(sendResponse, (e) => sendResponse({ ok: false, error: `${e.name || "Error"}: ${e.message}` }));
  return true;
});