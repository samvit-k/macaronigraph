"""Assemble the enriched record and decide its status (never a false success). Persistence: jobs.py."""
from __future__ import annotations

from datetime import datetime, timezone

from . import config

# Playback outcomes reported by the extension that make the item unusable.
FATAL_PLAYBACK = {"unavailable", "no_video", "auth_required", "playback_blocked", "capture_lost",
                  "capture_error", "navigation_error", "aborted"}
MIN_COVERAGE = 0.9
TRANSCRIPT_DONE = {"ok", "skipped"}   # "skipped" = measured silent; transcription deliberately not run
OCR_DONE = {"ok"}                     # "ok" with zero segments is a valid "no on-screen text" result


MUTED_MIN_S, MUTED_MAX_FRACTION = 1.0, 0.15


def muted_too_long(capture: dict) -> bool:
    """Muted long enough that speech may be missing. Legacy captures without a measured duration keep the
    strict rule (any mute → error)."""
    if not capture.get("muted"):
        return False
    muted_s = capture.get("muted_s")
    if muted_s is None:
        return True
    return muted_s > max(MUTED_MIN_S, MUTED_MAX_FRACTION * (capture.get("captured_s") or 0))


def warnings_for(capture: dict) -> list[str]:
    """Non-fatal observations recorded on the record (they never change the status)."""
    w = []
    if capture.get("muted") and capture.get("muted_s") is not None and not muted_too_long(capture):
        w.append(f"briefly_muted: {capture['muted_s']:.1f}s of {capture.get('captured_s') or 0:.1f}s")
    return w


def decide_status(capture: dict, transcript: dict | None, ocr: dict | None) -> tuple[str, list[str]]:
    """→ (status, errors). status ∈ ok | ok_no_speech | partial | failed.
    ok/ok_no_speech require: usable playback, ≥90% coverage, AND both extractors completed."""
    errors: list[str] = []
    playback = capture.get("playback") or "unknown"
    if playback in FATAL_PLAYBACK:
        detail = capture.get("playback_detail")
        return "failed", [f"{playback}: {detail}" if detail else playback]
    if playback == "stalled":
        errors.append("playback_stalled")
    elif playback != "ok":
        errors.append(f"unexpected_playback_state: {playback}")
    if capture.get("processing_error"):
        errors.append(f"processing_error: {capture['processing_error']}")
    if muted_too_long(capture):
        errors.append("audio_muted: page kept the video muted, so silence is not evidence of no speech"
                      + (f" ({capture['muted_s']:.1f}s muted)" if capture.get("muted_s") is not None else ""))

    expected = capture.get("expected_s") or 0
    captured = capture.get("captured_s") or 0
    full = capture.get("video_duration_s") or 0
    if expected <= 0:
        errors.append("unknown_video_duration")
    elif captured / expected < MIN_COVERAGE:
        errors.append(f"incomplete_capture: {captured:.1f}s of {expected:.1f}s")
    if capture.get("truncated") or (full and expected and full > expected + 0.5):
        errors.append(f"truncated_at_capture_limit: captured {captured:.1f}s of {full:.1f}s video "
                      f"(limit {capture.get('capture_limit_s') or expected:.0f}s)")

    t, o = transcript or {}, ocr or {}
    if transcript is None:
        errors.append("transcription_missing: no transcript result")
    elif t.get("status") not in TRANSCRIPT_DONE:
        errors.append(f"transcription_error: {t.get('error') or t.get('status')}")
    if ocr is None:
        errors.append("ocr_missing: no OCR result")
    elif o.get("status") not in OCR_DONE:
        errors.append(f"ocr_{o.get('status')}: {o.get('error') or 'no frames captured'}")

    has_content = bool(t.get("text") or t.get("uncertain_text") or o.get("segments"))
    if errors:
        return ("partial" if has_content else "failed"), errors
    if t.get("audio_class") == "speech":
        return "ok", []
    if t.get("audio_class") in ("silent", "non_speech"):
        return "ok_no_speech", []
    return "partial", [f"unclassified_audio: {t.get('audio_class')}"]


def passages(reel_id: str, url: str, caption: str, transcript: dict | None, ocr: dict | None) -> list[dict]:
    """Provenance-tagged text units (input for Phase 3 chunking/indexing; nothing is indexed here)."""
    out = []

    def add(kind: str, n: int, text: str, start=None, end=None, **extra):
        out.append({"chunk_id": f"{reel_id}:{kind}:{n:03d}", "reel_id": reel_id, "source_type": kind,
                    "start": start, "end": end, "text": text, "url": url, **extra})

    if caption.strip():
        add("caption", 0, caption.strip())
    for i, s in enumerate((transcript or {}).get("segments", [])):
        add("transcript", i, s["text"], s["start"], s["end"], confidence=s["confidence"])
    for i, s in enumerate((ocr or {}).get("segments", [])):
        add("ocr", i, s["text"], s["start"], s["end"], confidence=s["confidence"],
            ocr_kind=s.get("kind", "text"), ocr_parts=s.get("parts"),
            ocr_echo=(s.get("echo") or {}).get("label"), ocr_quality=s.get("quality", "normal"))
    return out


def build_record(post: dict | None, capture: dict, transcript: dict | None, ocr: dict | None) -> dict:
    reel_id = capture.get("reel_id") or (post or {}).get("reel_id")
    url = (post or {}).get("source_url") or capture.get("url")
    status, errors = decide_status(capture, transcript, ocr)
    if post is None:
        errors.append("metadata_not_found: reel_id not present in source JSON")
        status = "partial" if status in ("ok", "ok_no_speech") else status
    metadata = (post or {}).get("metadata", {})
    return {
        "reel_id": reel_id,
        "source_url": url,
        "saved_at": (post or {}).get("saved_at"),
        "source_index": (post or {}).get("source_index"),
        "session_id": capture.get("session_id"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": config.PIPELINE_VERSION,
        "status": status,
        "errors": errors,
        "warnings": warnings_for(capture),
        "metadata": metadata,
        "capture": {k: v for k, v in capture.items() if k not in ("reel_id", "url", "session_id")},
        "transcript": transcript,
        "ocr": ocr,
        "passages": passages(reel_id, url, metadata.get("caption", ""), transcript, ocr),
    }
