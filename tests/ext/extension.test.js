// node --test tests/ext
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { cropPx } = require("../../extension/crop.js");

const EXT = path.join(__dirname, "../../extension");

test("crop: stream = viewport × dpr maps 1:1 after scaling", () => {
  // 1280x800 CSS viewport @2x, stream 2560x1600; video centered 450 CSS px wide
  const c = cropPx(2560, 1600, { w: 1280, h: 800 }, { x: 415, y: 0, w: 450, h: 800 });
  assert.deepEqual(c, { x: 830, y: 0, w: 900, h: 1600 });
});

test("crop: letterboxed stream (aspect differs) is offset and scaled", () => {
  // viewport 1000x500 (2:1) into 1000x1000 stream → scale 1, 250 px bars top/bottom
  const c = cropPx(1000, 1000, { w: 1000, h: 500 }, { x: 100, y: 50, w: 200, h: 100 });
  assert.deepEqual(c, { x: 100, y: 300, w: 200, h: 100 });
  // pillarboxed: viewport 500x1000 into 1000x1000 → scale 1, 250 px bars left/right
  assert.deepEqual(cropPx(1000, 1000, { w: 500, h: 1000 }, { x: 0, y: 0, w: 500, h: 1000 }),
                   { x: 250, y: 0, w: 500, h: 1000 });
});

test("crop: rect partly outside the viewport is clamped to the frame", () => {
  const c = cropPx(1280, 800, { w: 1280, h: 800 }, { x: -100, y: 700, w: 400, h: 300 });
  assert.deepEqual(c, { x: 0, y: 700, w: 300, h: 100 });
});

test("crop: downscaled stream (maxWidth constraint) uses actual frame size", () => {
  const c = cropPx(1280, 800, { w: 1280, h: 800 }, { x: 640, y: 400, w: 640, h: 400 });
  assert.deepEqual(c, { x: 640, y: 400, w: 640, h: 400 });
  const d = cropPx(640, 400, { w: 1280, h: 800 }, { x: 640, y: 400, w: 640, h: 400 });
  assert.deepEqual(d, { x: 320, y: 200, w: 320, h: 200 });
});

// ---- background.js with a mocked chrome API ----
function loadBackground({ failAt }) {
  const calls = [];
  let offscreenOpen = false;
  const chrome = {
    runtime: {
      onMessage: { addListener() {} },
      getContexts: async () => (offscreenOpen ? [{}] : []),
      sendMessage: async (msg) => {
        calls.push(`offscreen:${msg.type}`);
        if (msg.type === "start-capture") return failAt === "start-capture" ? { ok: false, error: "NotAllowedError" } : { ok: true };
        return { ok: true };
      },
    },
    offscreen: {
      createDocument: async () => { calls.push("createDocument"); offscreenOpen = true; },
      closeDocument: async () => { calls.push("closeDocument"); offscreenOpen = false; },
    },
    tabs: { get: async () => ({ url: "http://localhost:8765/" }), onUpdated: { addListener() {}, removeListener() {} } },
    scripting: { executeScript: async () => [{ result: { w: 100, h: 100, dpr: 1 } }] },
    tabCapture: {
      getMediaStreamId: async () => {
        calls.push("getMediaStreamId");
        if (failAt === "getMediaStreamId") throw new Error("Extension has not been invoked for the current page");
        return "sid";
      },
    },
    storage: { session: { set: async () => {} } },
  };
  const fetch = async (url) => { calls.push(`fetch:${url.split("?")[0].replace("http://localhost:8765", "")}`);
    return { ok: true, json: async () => ({ session_id: "s1", items: [{ reel_id: "A", url: "http://localhost:8765/x" }] }) }; };
  const ctx = { chrome, fetch, console: { log() {} }, setTimeout, clearTimeout, module: { exports: {} } };
  vm.runInNewContext(fs.readFileSync(path.join(EXT, "background.js"), "utf8"), ctx);
  return { bg: ctx.module.exports, calls, isOpen: () => offscreenOpen };
}

for (const failAt of ["getMediaStreamId", "start-capture"]) {
  test(`start(): failure at ${failAt} stops capture, closes offscreen, allows retry`, async () => {
    const { bg, calls, isOpen } = loadBackground({ failAt });
    await assert.rejects(bg.start({ tabId: 1, source: "test", limit: 3, offset: 0, intervalMs: 1000 }));
    assert.ok(calls.includes("offscreen:stop-capture"), calls.join(","));
    assert.ok(calls.includes("closeDocument"), calls.join(","));
    assert.equal(isOpen(), false);
    assert.equal(bg._state(), null);
    assert.ok(calls.includes("fetch:/api/session/s1/finish"), "leased Reels must be released: " + calls.join(","));
    // a second attempt is not blocked by stale state
    await assert.rejects(bg.start({ tabId: 1, source: "test", limit: 3, offset: 0, intervalMs: 1000 }),
                         (e) => !/already running/.test(e.message));
  });
}