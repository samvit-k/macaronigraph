// Session orchestrator. One user click (popup "Start Processing") → capture the active tab once,
// then navigate that same tab through the queue, recording each item. Per-item failures are recorded
// and the session continues; auth walls and capture loss stop the session (never bypassed).
const API = "http://localhost:8765";
const MAX_CAPTURE_S = 180;
const STALL_MS = 6000;
const NAV_TIMEOUT_MS = 30000;
const PACE_MS = { test: 1000, instagram: 5000 };

let S = null;          // current session state
let starting = false;  // guards concurrent Start clicks during initialization

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function publish(patch = {}) {
  if (!S) return;
  Object.assign(S.view, patch);
  chrome.storage.session.set({ state: S.view });
}
function log(line) {
  console.log("[reel-capture]", line);
  if (!S) return;
  S.view.log = [...S.view.log, `${new Date().toLocaleTimeString()}  ${line}`].slice(-200);
  publish();
}

async function ensureOffscreen() {
  const existing = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"] });
  if (existing.length) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html", reasons: ["USER_MEDIA"],
    justification: "Record tab audio and sample frames of the user-started processing session",
  });
}
const offscreen = (type, payload = {}) => chrome.runtime.sendMessage({ target: "offscreen", type, ...payload });

// Stop any (partially) started capture and close the offscreen document. Safe to call repeatedly.
async function teardown() {
  await offscreen("stop-capture").catch(() => {});
  const ctx = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"] }).catch(() => []);
  if (ctx.length) await chrome.offscreen.closeDocument().catch(() => {});
}

async function call(tabId, method, ...args) {
  const [res] = await chrome.scripting.executeScript({
    target: { tabId },
    func: (m, a) => window.__reelCap[m](...a),
    args: [method, args],
  });
  return res?.result;
}
async function inject(tabId) {
  await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
}

// Same-origin items are opened from inside the page (renderer-initiated) so Chrome can carry the user's
// in-page click forward; cross-origin items fall back to tabs.update (activation will not carry).
async function navigate(tabId, url) {
  const tab = await chrome.tabs.get(tabId);
  const sameOrigin = (() => { try { return new URL(tab.url).origin === new URL(url).origin; } catch { return false; } })();
  return waitForLoad(tabId, sameOrigin
    ? () => call(tabId, "navigateTo", url)
    : () => chrome.tabs.update(tabId, { url }));
}

function waitForLoad(tabId, trigger) {
  return new Promise((resolve, reject) => {
    let sawLoading = false;
    const timer = setTimeout(() => { done(); reject(new Error(`navigation timeout (${NAV_TIMEOUT_MS} ms)`)); }, NAV_TIMEOUT_MS);
    const listener = (id, info, tab) => {
      if (id !== tabId) return;
      if (info.status === "loading") sawLoading = true;
      if (sawLoading && info.status === "complete") { done(); resolve(tab); }
    };
    const done = () => { clearTimeout(timer); chrome.tabs.onUpdated.removeListener(listener); };
    chrome.tabs.onUpdated.addListener(listener);
    Promise.resolve().then(trigger).catch((e) => { done(); reject(e); });
  });
}

// Job-store bookkeeping (Phase 2): keep this session's leases alive, and release them when it ends.
const HEARTBEAT_MS = 30000;
function finishSession(sessionId, reason) {
  return fetch(`${API}/api/session/${sessionId}/finish?reason=${encodeURIComponent(reason)}`, { method: "POST" })
    .catch(() => {}); // backend reclaims stale leases itself if this never arrives
}

async function start({ tabId, source, limit, offset, intervalMs, mode = "pending" }) {
  if (S?.running || starting) throw new Error("a session is already running");
  starting = true;
  let leased = null;
  try {
    const tab = await chrome.tabs.get(tabId);
    if (!/^https?:/.test(tab.url || "")) {
      throw new Error("open the backend start page (http://localhost:8765/) or an http(s) page first");
    }
    // Fetch the queue BEFORE requesting the stream id, so the id is consumed immediately after issue.
    const sres = await fetch(`${API}/api/session?source=${source}&limit=${limit}&offset=${offset}&mode=${mode}`);
    if (!sres.ok) throw new Error(`backend /api/session: ${sres.status} ${await sres.text()}`);
    const session = await sres.json();
    if (!session.items.length) {
      await finishSession(session.session_id, "empty queue");
      const label = { test: "test media", instagram: "instagram saved" }[source] || source;
      throw new Error(
        mode === "pending" ? `nothing pending for source "${label}": every Reel there is done ` +
                             `(check the Source selector, or choose 'reprocess all' to redo)`
        : mode === "eval" ? `no OCR eval set enabled for source "${label}" (python -m backend.evalframes enable …)`
        : `queue is empty for source "${label}"`);
    }
    leased = session.session_id;

    await ensureOffscreen();
    await inject(tabId);
    const viewport = await call(tabId, "viewport");
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
    const cap = await offscreen("start-capture", { streamId, viewport });
    if (!cap?.ok) throw new Error(`tab capture failed: ${cap?.error}`);

    S = { running: true, armed: false, abort: false, captureLost: null, tabId, source, intervalMs, session,
          view: { running: true, session_id: session.session_id, source, total: session.items.length,
                  done: 0, current: "waiting for the ▶ Start click on the page", results: [], log: [] } };
    log(`capture started: ${cap.audioTracks} audio / ${cap.videoTracks} video track(s), ` +
        `${cap.settings?.width}x${cap.settings?.height}`);
    const firstOrigin = new URL(session.items[0].url).origin;
    if (new URL(tab.url).origin !== firstOrigin) {
      log(`warning: this tab is not on ${firstOrigin}; the page click will not carry over. ` +
          `Open ${firstOrigin}/ in this tab and start again.`);
    }
    await call(tabId, "arm");
    log("click ▶ Start Processing on the page (Chrome requires an in-page click before playing sound)");
  } catch (e) {
    await teardown();
    if (leased) await finishSession(leased, `init failed: ${e.message}`);
    S = null;
    throw e;
  } finally {
    starting = false;
  }
  const mine = S;
  setTimeout(() => {
    if (S === mine && !mine.armed && mine.running) {
      log("no in-page click within 2 minutes; stopping capture");
      call(tabId, "disarm").catch(() => {});
      finishSession(mine.session.session_id, "not armed");
      teardown().then(() => { mine.running = false; publish({ running: false, current: null }); });
    }
  }, 120000);
  return { ok: true, session_id: S.session.session_id, items: S.session.items.length };
}

function onArmed(msg, sender) {
  if (!S || S.armed || !S.running || sender.tab?.id !== S.tabId) return;
  if (!msg.trusted) { log("ignored untrusted (synthetic) click"); return; }
  S.armed = true;
  log(`in-page click received (userActivation.hasBeenActive=${msg.activation?.hasBeenActive}); starting`);
  S.done = run().catch((e) => log(`session crashed: ${e.message}`));
}

async function run() {
  const items = S.session.items;
  const sid = S.session.session_id;
  let endReason = "finished";
  const hb = setInterval(() => fetch(`${API}/api/session/${sid}/heartbeat`, { method: "POST" }).catch(() => {}),
                         HEARTBEAT_MS);
  try {
    for (const [i, it] of items.entries()) {
      if (S.abort) { log("stopped by user"); endReason = "stopped by user"; break; }
      publish({ current: `${i + 1}/${items.length} ${it.reel_id}` });
      const out = await processItem(it, i);
      S.view.results = [...S.view.results, { reel_id: it.reel_id, ...out }];
      publish({ done: i + 1 });
      log(`${it.reel_id}: ${out.status}${out.state ? ` [${out.state}]` : ""}` +
          `${out.errors?.length ? " — " + out.errors.join("; ") : ""}`);
      if (out.stopSession) { log(`session stopped: ${out.stopSession}`); endReason = out.stopSession; break; }
      if (i < items.length - 1) await sleep(PACE_MS[S.source] ?? 3000);
    }
  } catch (e) {
    endReason = `crashed: ${e.message}`;
    throw e;
  } finally {
    await teardown();                     // capturing is over: release the tab right away
    await finishSession(sid, endReason);  // un-uploaded leased Reels go straight back to the queue
    log("capture stopped; waiting for the backend to finish processing");
    await waitForProcessing(sid);         // heartbeats continue: the session is not final until drained
    clearInterval(hb);
    S.running = false;
    publish({ running: false, current: null });
    log("session finished");
  }
}

// The backend processes uploads in the background. Keep the session alive (heartbeats run in run())
// until none of its uploads are queued, then report the final per-Reel outcomes.
const DRAIN_POLL_MS = 5000;
const DRAIN_MAX_MS = 60 * 60 * 1000;
async function waitForProcessing(sid) {
  const t0 = Date.now();
  let last = -1;
  while (Date.now() - t0 < DRAIN_MAX_MS) {
    let st;
    try { st = await (await fetch(`${API}/api/session/${sid}/status`)).json(); }
    catch (e) { log(`backend unreachable while draining (${e.message}); it will resume processing on restart`); return; }
    if (!st.queued) break;
    if (st.queued !== last) { publish({ current: `processing ${st.queued} queued upload(s)` }); last = st.queued; }
    await sleep(DRAIN_POLL_MS);
  }
  try {
    const rows = await (await fetch(`${API}/api/results?session_id=${sid}`)).json();
    for (const r of rows) {
      log(`result ${r.reel_id}: ${r.status} [${r.outcome}]${r.errors?.length ? " — " + r.errors.join("; ") : ""}`);
    }
    const st = await (await fetch(`${API}/api/session/${sid}/status`)).json();
    if (st.queued) log(`warning: ${st.queued} upload(s) still processing in the backend; results will follow`);
  } catch (e) {
    log(`could not fetch final results: ${e.message}`);
  }
}

// A backend 503 means its processing backlog stayed full: stop instead of burning through the queue.
function failed(r) {
  const error = r?.error || "unknown error";
  return { status: "failed", errors: [error], ...(/backend 503/.test(error) ? { stopSession: "backend overloaded" } : {}) };
}

async function processItem(it, index) {
  const tabId = S.tabId;
  const base = { reel_id: it.reel_id, url: it.url, session_id: S.session.session_id, source: S.source,
                 item_index: index, capture_limit_s: MAX_CAPTURE_S };
  const metaOnly = async (playback, detail, extra = {}) => {
    const r = await offscreen("post-meta", { meta: { ...base, playback, playback_detail: detail, ...extra } });
    return r?.ok ? r.result : failed(r);
  };

  if (S.captureLost) return { ...(await metaOnly("capture_lost", S.captureLost)), stopSession: "capture lost" };
  let tab;
  try { tab = await navigate(tabId, it.url); }
  catch (e) { return metaOnly("navigation_error", e.message); }
  if (/instagram\.com\/accounts\/login/.test(tab.url || "")) {
    return { ...(await metaOnly("auth_required", `redirected to ${tab.url}`)), stopSession: "Instagram login required" };
  }

  let prep;
  try { await inject(tabId); prep = await call(tabId, "prepare"); }
  catch (e) { return metaOnly("capture_error", `content script: ${e.message}`); }
  if (prep?.status !== "ready") {
    const r = await metaOnly(prep?.status || "capture_error", prep?.detail, { final_url: tab.url });
    return prep?.status === "auth_required" ? { ...r, stopSession: "login required" } : r;
  }

  // expected_s = what this capture intends to cover; video_duration_s = the real full length.
  const truncated = prep.duration > MAX_CAPTURE_S;
  const expected = truncated ? MAX_CAPTURE_S : prep.duration;
  const b = await offscreen("begin-item", { intervalMs: S.intervalMs, crop: { rect: prep.rect, viewport: prep.viewport } });
  if (!b?.ok) return { ...(await metaOnly("capture_lost", b?.error)), stopSession: "capture stream unavailable" };

  const play = await call(tabId, "play").catch((e) => ({ status: "capture_error", detail: e.message }));
  const meta = { ...base, final_url: prep.url, video_duration_s: prep.duration, expected_s: expected, truncated,
                 viewport: prep.viewport, video_rect: prep.rect, user_activation: prep.activation };
  if (play?.status !== "ok") {
    const r = await offscreen("end-item", { upload: false, meta: { ...meta, playback: play?.status, playback_detail: play?.detail } });
    return r?.ok ? r.result : failed(r);
  }

  // Monitor until the video ends, loops, stalls, or the cap is reached.
  let reason = "ok", detail = null, maxT = 0, lastT = -1, lastProgress = Date.now(), mutedSeen = !!play.muted;
  let mutedMs = 0, lastPoll = Date.now();  // how long the page kept the video muted (brief ≠ unreliable)
  const deadline = Date.now() + (expected + 15) * 1000;
  while (true) {
    await sleep(400);
    if (S.abort) { reason = "aborted"; detail = "stopped by user"; break; }
    if (S.captureLost) { reason = "capture_lost"; detail = S.captureLost; break; }
    let m;
    try { m = await call(tabId, "monitor"); } catch (e) { reason = "capture_error"; detail = `page changed: ${e.message}`; break; }
    if (!m || m.gone) { reason = "capture_error"; detail = "video element disappeared"; break; }
    const now = Date.now();
    if (m.muted) mutedMs += now - lastPoll;
    lastPoll = now;
    mutedSeen ||= m.muted;
    if (m.t + 0.5 < lastT && lastT > expected * 0.8) { maxT = Math.max(maxT, lastT); break; } // page looped it
    if (m.t > lastT + 0.01) lastProgress = Date.now();
    lastT = m.t; maxT = Math.max(maxT, m.t);
    if (m.ended) { maxT = prep.duration; break; }
    if (m.t >= expected - 0.1) break;
    if (Date.now() - lastProgress > STALL_MS) { reason = "stalled"; detail = `no progress at ${m.t.toFixed(1)}s`; break; }
    if (Date.now() > deadline) { reason = "stalled"; detail = "capture deadline exceeded"; break; }
  }
  await call(tabId, "stop").catch(() => {});

  const r = await offscreen("end-item", { upload: true, meta: { ...meta, playback: reason, playback_detail: detail,
    muted: mutedSeen, muted_s: Math.round(mutedMs / 100) / 10, unmute: play.unmute, play_wall: play.playWall,
    captured_s: Math.round(Math.min(maxT, expected) * 100) / 100 } });
  if (!r?.ok) return failed(r);
  log(`${it.reel_id}: frames ${r.meta.frames_sampled}/${r.meta.frames_attempted} ` +
      `(interval ${r.meta.frame_interval_actual_ms} ms), audio ${r.meta.audio_bytes_client} B, ` +
      `peak ${r.meta.audio_peak_db} dB, stream_live=${r.meta.stream_live}`);
  const out = r.result;
  return reason === "capture_lost" ? { ...out, stopSession: "capture lost" } : out;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.target !== "background") return;
  if (msg.type === "start") {
    start(msg).then(sendResponse, (e) => sendResponse({ ok: false, error: e.message }));
    return true;
  }
  if (msg.type === "armed") onArmed(msg, _sender);
  if (msg.type === "stop") {
    if (S) S.abort = true;
    if (S && !S.armed && S.running) {  // stopped before the in-page click: nothing is running yet, so clean up here
      call(S.tabId, "disarm").catch(() => {});
      finishSession(S.session.session_id, "stopped before start");
      teardown().then(() => { S.running = false; publish({ running: false, current: null }); });
    }
    sendResponse({ ok: true });
  }
  if (msg.type === "capture-lost") { if (S) { S.captureLost = msg.detail; log(`capture lost: ${msg.detail}`); } }
});

if (typeof module !== "undefined") module.exports = { start, teardown, _state: () => S };
