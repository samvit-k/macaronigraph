// Pure geometry: map the video's CSS rect (tab viewport coords) to pixels in the captured frame.
// Tab capture scales the viewport into the stream frame preserving aspect ratio; if the stream's
// aspect differs (constraints), the content is letterboxed and centered. Uses the actual frame size.
function cropPx(frameW, frameH, viewport, rect) {
  const s = Math.min(frameW / viewport.w, frameH / viewport.h);
  const ox = (frameW - viewport.w * s) / 2;
  const oy = (frameH - viewport.h * s) / 2;
  const x0 = Math.max(0, Math.round(ox + rect.x * s));
  const y0 = Math.max(0, Math.round(oy + rect.y * s));
  const x1 = Math.min(frameW, Math.round(ox + (rect.x + rect.w) * s));
  const y1 = Math.min(frameH, Math.round(oy + (rect.y + rect.h) * s));
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}
if (typeof module !== "undefined") module.exports = { cropPx };
