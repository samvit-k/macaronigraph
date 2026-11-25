// Extension popup: start/stop messages to background.js, session state from chrome.storage.session, library stats from the backend.
const API = "http://localhost:8765";
const $ = (id) => document.getElementById(id);
const FIELDS = ["source", "limit", "offset", "interval", "mode"];
const MIN_PER_REEL = 1.05;

function syncSeg(seg) {
  const input = $(seg.dataset.for);
  for (const b of seg.querySelectorAll("button")) {
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(b.dataset.value === input.value));
  }
}
const syncControls = () => {
  document.querySelectorAll(".seg").forEach(syncSeg);
  $("evalToggle").checked = $("mode").value === "eval";
  updateEstimate();
};
for (const seg of document.querySelectorAll(".seg")) {
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $(seg.dataset.for).value = b.dataset.value;
    syncControls();
    $(seg.dataset.for).dispatchEvent(new Event("change"));
  });
}
$("evalToggle").addEventListener("change", () => {
  $("mode").value = $("evalToggle").checked ? "eval" : "pending";
  syncControls();
  $("mode").dispatchEvent(new Event("change"));
});
for (const b of document.querySelectorAll(".stepper button")) {
  b.addEventListener("click", () => {
    const i = $("limit");
    i.value = Math.max(1, Math.min(900, (Number(i.value) || 0) + Number(b.dataset.step)));
    i.dispatchEvent(new Event("change"));
  });
}
function updateEstimate() {
  const n = Number($("limit").value) || 0;
  const min = Math.round(n * MIN_PER_REEL);
  $("estimate").textContent = n ? `${min < 60 ? `${min} Min` : `${(min / 60).toFixed(1)} Hours`} · ` : "";
}

const saveForm = () => chrome.storage.local.set({ form: Object.fromEntries(FIELDS.map((f) => [f, $(f).value])) });
FIELDS.forEach((f) => $(f).addEventListener("change", () => { saveForm(); syncControls(); refreshStats(); }));
$("limit").addEventListener("input", updateEstimate);
chrome.storage.local.get("form").then(({ form }) => {
  for (const f of FIELDS) if (form?.[f] !== undefined) $(f).value = form[f];
  syncControls();
  refreshStats();
});
syncControls();

async function refreshHealth() {
  const h = $("health");
  try {
    const r = await (await fetch(`${API}/api/health`)).json();
    h.className = "status ok";
    $("healthText").textContent = r.backlog ? `Processing ${r.backlog}` : "Online";
    const model = String(r.whisper_model || "").split("/").pop();
    h.title = `Backend Online · ${model}${r.quarantined ? ` · ${r.quarantined} Quarantined Job(s)` : ""}`;
  } catch {
    h.className = "status bad";
    $("healthText").textContent = "Offline";
    h.title = "Backend Not Reachable On localhost:8765 (Start It With uvicorn)";
  }
}

const pct = (part, whole) => (whole ? `${(100 * part) / whole}%` : "0");
async function refreshStats() {
  try {
    const s = await (await fetch(`${API}/api/status`)).json();
    const st = s.reels?.[$("source").value] || {};
    const n = (k) => st[k] || 0;
    const total = Object.entries(st).filter(([k]) => k !== "skipped").reduce((a, [, v]) => a + v, 0);
    const done = n("done");
    const active = n("in_progress") + n("queued");
    const failed = n("needs_retry") + n("exhausted") + n("failed_permanent");
    $("stDone").textContent = done;
    $("stTotal").textContent = total;
    $("lgDone").textContent = done;
    $("lgPending").textContent = n("pending");
    $("lgFailed").textContent = failed;
    $("lgFailedWrap").title = `${n("needs_retry")} Will Be Retried · ${n("exhausted")} Out Of Retries · ` +
                              `${n("failed_permanent")} Unavailable`;
    $("barDone").style.width = pct(done, total);
    $("barActive").style.width = pct(active, total);
    $("barFailed").style.width = pct(failed, total);
    const notes = [];
    if (active) notes.push(`${active} In The Current Session`);
    if (n("needs_retry")) notes.push(`${n("needs_retry")} Queued For Retry`);
    $("libNote").textContent = notes.join(" · ");
    $("ofTotal").title = n("skipped") ? `${n("skipped")} Non-Reel Posts (Images, Carousels) Skipped` : "";
    fitLog();
  } catch {
    for (const id of ["stDone", "stTotal", "lgDone", "lgPending", "lgFailed"]) $(id).textContent = "–";
    $("libNote").textContent = "";
  }
}

const LINE_RE = /^(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s{2}([\s\S]*)$/;
const MARKS = [
  { re: /\b(failed|crashed|capture lost|overloaded|unreachable|offline|login required|error)\b/i, cls: "bad", label: "×" },
  { re: /\b(partial|needs_retry|warning|briefly_muted)\b/i, cls: "warn", label: "!" },
  { re: /\b(ok|ok_no_speech|done|session finished)\b/, cls: "ok", label: "✓" },
  { re: /\b(queued|capture started|waiting|in-page click|click ▶)/, cls: "info", label: "•" },
];
function renderLogLine(line) {
  const li = document.createElement("li");
  const m = LINE_RE.exec(line);
  const text = m ? m[2] : line;
  const time = document.createElement("time");
  time.textContent = m ? m[1].replace(/\s?[AP]M$/, "") : "";
  const mark = document.createElement("span");
  const hit = MARKS.find((x) => x.re.test(text));
  mark.className = `mark ${hit ? hit.cls : ""}`;
  mark.textContent = hit ? hit.label : "";
  const msg = document.createElement("span");
  msg.className = "msg";
  msg.textContent = text;
  li.append(time, mark, msg);
  return li;
}

// Chrome caps popups at 600px: the activity log takes the remaining height so the page never scrolls.
const POPUP_MAX = 600;
function fitLog() {
  const log = $("log");
  if (!$("logWrap").open) return;
  log.style.maxHeight = "none";
  const others = document.documentElement.scrollHeight - log.offsetHeight;
  log.style.maxHeight = `${Math.max(48, Math.floor(POPUP_MAX - others) - 2)}px`;
  log.scrollTop = log.scrollHeight;
}
$("logWrap").addEventListener("toggle", fitLog);
document.querySelectorAll("details").forEach((d) => d.addEventListener("toggle", fitLog));

let wasRunning = null;
function render(state) {
  const lines = (state?.log || []).slice(-60);
  const log = $("log");
  if (lines.length) log.replaceChildren(...lines.map(renderLogLine));
  else log.replaceChildren(Object.assign(document.createElement("li"), { className: "empty", textContent: "No Activity Yet" }));
  log.scrollTop = log.scrollHeight;
  $("logCount").textContent = state?.log?.length ? String(state.log.length) : "";

  const running = Boolean(state?.running);
  $("idleView").hidden = running;
  $("runView").hidden = !running;
  $("start").disabled = running;
  $("stop").disabled = !running;
  if (running) {
    $("runDone").textContent = state.done || 0;
    $("runTotal").textContent = state.total || 0;
    $("runBar").style.width = pct(state.done || 0, state.total || 0);
    const current = String(state.current || "").replace(/^\d+\/\d+\s+/, "");
    const cur = $("runCurrent");
    cur.replaceChildren();
    if (/^processing|waiting/i.test(current)) cur.append(current);
    else if (current) cur.append("Now: ", Object.assign(document.createElement("b"), { textContent: current }));
  }
  if (wasRunning !== running && running) $("logWrap").open = true;
  wasRunning = running;
  fitLog();
}

chrome.storage.session.get("state").then(({ state }) => render(state));
chrome.storage.onChanged.addListener((ch, area) => {
  if (area === "session" && ch.state) { render(ch.state.newValue); refreshStats(); }
});

function showHint(text, isError = false) {
  $("hint").className = isError ? "hint error" : "hint";
  $("hintText").textContent = text;
}

$("start").onclick = async () => {
  $("start").disabled = true;
  let started = false;
  try {
    await saveForm();
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) throw new Error("No Active Tab Found: Open The Popup From The Worker Tab");
    const res = await chrome.runtime.sendMessage({
      target: "background", type: "start", tabId: tab.id, source: $("source").value,
      limit: Number($("limit").value), offset: Number($("offset").value),
      intervalMs: Number($("interval").value), mode: $("mode").value,
    });
    if (!res?.ok) throw new Error(res?.error || "The Extension Did Not Respond: Try Reloading It");
    started = true;
    showHint("Now Click ▶ Start Processing On The Page To Allow Sound");
  } catch (e) {
    showHint(e?.message || String(e), true);
  } finally {
    if (!started) $("start").disabled = false;
  }
};

$("stop").onclick = async () => {
  $("stop").disabled = true;
  try {
    await chrome.runtime.sendMessage({ target: "background", type: "stop" });
    $("runTitle").textContent = "Stopping…";
  } catch (e) {
    $("stop").disabled = false;
    $("runCurrent").textContent = `Could Not Stop: ${e?.message || e}`;
  }
};

refreshHealth();
refreshStats();
setInterval(() => { refreshHealth(); refreshStats(); }, 5000);