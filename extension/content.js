// Injected into the worker tab: reads the page and controls only its <video>; never logs in or works around access restrictions.
(() => {
  if (window.__reelCap) return;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function pageBlock() {
    const text = `${document.title}\n${(document.body && document.body.innerText) || ""}`.slice(0, 5000);
    if (/\/accounts\/login|\/challenge\//.test(location.pathname) ||
        document.querySelector('input[name="password"]')) {
      return { status: "auth_required", detail: `login required at ${location.pathname}` };
    }
    if (/Sorry, this page isn.t available|Page not found|content isn.t available|no longer available/i.test(text)) {
      return { status: "unavailable", detail: document.title || "page reports content unavailable" };
    }
    if (/try again later|please wait a few minutes/i.test(text)) {
      return { status: "unavailable", detail: "rate limited by site" };
    }
    return null;
  }

  function pickVideo() {
    let best = null, bestArea = 0;
    for (const v of document.querySelectorAll("video")) {
      const r = v.getBoundingClientRect();
      const w = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
      const h = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
      if (w * h > bestArea) { best = v; bestArea = w * h; }
    }
    return best;
  }

  // Walks up the video's ancestors, so another Reel's sound toggle in a feed is never chosen.

  const SOUND_RE = /audio is muted|toggle audio|unmute|sound off|turn on sound/i;
  function findSoundToggle(v) {
    let node = v.parentElement;
    for (let depth = 0; node && depth < 10; depth++, node = node.parentElement) {
      for (const el of node.querySelectorAll("[aria-label], svg[aria-label], title")) {
        const label = el.getAttribute?.("aria-label") || el.textContent || "";
        if (SOUND_RE.test(label)) return el.closest('button, [role="button"]') || el;
      }
    }
    return null;
  }

  const activation = () => ({ hasBeenActive: navigator.userActivation?.hasBeenActive ?? null,
                              isActive: navigator.userActivation?.isActive ?? null });

  window.__reelCap = {
    viewport() {
      return { w: innerWidth, h: innerHeight, dpr: devicePixelRatio };
    },

    // Chrome permits unmuted playback only after a real click in the page; the session starts from that click.

    arm() {
      document.getElementById("__reelcap_arm")?.remove();
      const b = document.createElement("button");
      b.id = "__reelcap_arm";
      b.textContent = "▶ Start Processing (click to allow sound)";
      b.style.cssText = "position:fixed;z-index:2147483647;top:50%;left:50%;transform:translate(-50%,-50%);" +
        "padding:18px 28px;font:600 20px system-ui;background:#0a7cff;color:#fff;border:0;border-radius:12px;" +
        "box-shadow:0 8px 30px rgba(0,0,0,.4);cursor:pointer";
      b.addEventListener("click", (e) => {
        b.remove();
        chrome.runtime.sendMessage({ target: "background", type: "armed", trusted: e.isTrusted, activation: activation() });
      }, { once: true });
      document.documentElement.appendChild(b);
      return true;
    },

    disarm() {
      document.getElementById("__reelcap_arm")?.remove();
      return true;
    },

    // Renderer-initiated same-origin navigation lets Chrome carry the page's user activation forward.
    navigateTo(url) {
      setTimeout(() => location.assign(url), 0);
      return true;
    },

    activation,

    async prepare(timeoutMs = 15000) {
      const t0 = Date.now();
      let v = null;
      while (Date.now() - t0 < timeoutMs) {
        const block = pageBlock();
        if (block) return block;
        v = pickVideo();
        if (v && v.readyState >= 1 && Number.isFinite(v.duration) && v.duration > 0) break;
        v = null;
        await sleep(250);
      }
      if (!v) return pageBlock() || { status: "no_video", detail: `no playable <video> within ${timeoutMs} ms` };
      v.pause();
      v.loop = false;
      v.currentTime = 0;
      window.__reelCapVideo = v;
      const r = v.getBoundingClientRect();
      return {
        status: "ready", url: location.href, duration: v.duration, activation: activation(),
        rect: { x: r.left, y: r.top, w: r.width, h: r.height },
        viewport: { w: innerWidth, h: innerHeight, dpr: devicePixelRatio },
      };
    },

    // If the browser's autoplay policy refuses unmuted playback, report it; no workaround.
    async play() {
      const v = window.__reelCapVideo;
      if (!v || !v.isConnected) return { status: "no_video", detail: "video element disappeared" };
      v.muted = false;
      v.volume = 1;
      try {
        await Promise.race([v.play(), sleep(5000).then(() => { throw new Error("play() timed out"); })]);
      } catch (e) {
        return { status: "playback_blocked", detail: `${e.name || "Error"}: ${e.message}`, activation: activation() };
      }
      const start = v.currentTime;
      await sleep(700);
      if (v.paused) return { status: "playback_blocked", detail: "video paused itself after play() (autoplay policy?)" };
      if (v.currentTime <= start) return { status: "playback_blocked", detail: "playback not progressing" };
      let unmute = v.muted ? "failed" : "direct";
      if (v.muted) {
        // The site re-muted it via its own player state: press its own sound toggle for this video.
        const toggle = findSoundToggle(v);
        if (toggle) {
          toggle.click();
          await sleep(800);
          if (v.muted) { v.muted = false; await sleep(300); }
          unmute = v.muted ? "failed" : "site_toggle";
          if (!v.muted) { v.currentTime = 0; await sleep(250); }  // don't lose the muted first ~1.5 s
        } else {
          unmute = "failed: no sound toggle found";
        }
      }
      return { status: "ok", playWall: Date.now() - v.currentTime * 1000, muted: v.muted || v.volume === 0, unmute };
    },

    monitor() {
      const v = window.__reelCapVideo;
      if (!v || !v.isConnected) return { gone: true };
      return { t: v.currentTime, paused: v.paused, ended: v.ended, muted: v.muted || v.volume === 0,
               loop: v.loop, wall: Date.now() };
    },

    stop() {
      const v = window.__reelCapVideo;
      if (v) v.pause();
      return true;
    },
  };
})();