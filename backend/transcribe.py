"""Audio → 16 kHz mono PCM (ffmpeg) → silence check → MLX-Whisper → per-segment confidence tiers.

A successful Whisper call is NOT proof of speech. Each segment is graded with Whisper's own signals:
  confident  – kept in `text`
  uncertain  – kept in `segments` with confidence="low" + flags (never silently dropped)
  rejected   – only clear non-content: empty/symbol-only output, repetition loops, or Whisper's
               combined silence rule (high no_speech_prob AND low avg_logprob). Listed in `rejected_segments`.
audio_class: silent | non_speech (audible, no confident segment) | speech | missing | unknown (whisper error)
"""
from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path

import numpy as np

from . import config

SR = 16000
SILENT_PEAK_DB = -50.0       # below this the capture carried no meaningful audio
NO_SPEECH_PROB_MAX = 0.6     # Whisper's defaults
AVG_LOGPROB_MIN = -1.0
COMPRESSION_RATIO_MAX = 2.4
# Phrases Whisper characteristically emits on music/silence. They are often *also* real speech,
# so a match only downgrades a segment to "uncertain"; it never deletes it.
HALLUCINATION_PHRASES = {
    "thank you", "thanks for watching", "thank you for watching", "thank you so much for watching",
    "please subscribe", "subscribe to my channel", "like and subscribe", "you", "bye",
    "subtitles by the amara org community", "transcription by castingwords",
}
_whisper_lock = threading.Lock()  # MLX/Metal: one transcription at a time


class AudioDecodeError(RuntimeError):
    pass


def norm(s: str) -> str:
    """Unicode-aware normalization: casefold, collapse non-word chars (\\w covers all scripts)."""
    return re.sub(r"[\W_]+", " ", s.casefold()).strip()


def decode_pcm(path: Path) -> np.ndarray:
    """Any container/codec ffmpeg understands → float32 mono 16 kHz. Raises AudioDecodeError."""
    proc = subprocess.run([config.FFMPEG, "-v", "error", "-nostdin", "-i", str(path), "-vn",
                           "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    if proc.returncode != 0:
        raise AudioDecodeError(proc.stderr.decode(errors="replace").strip()[-400:] or "ffmpeg failed")
    pcm = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        raise AudioDecodeError("no audio samples decoded (missing audio track?)")
    return pcm


def levels(pcm: np.ndarray) -> dict:
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    rms = float(np.sqrt(np.mean(pcm ** 2))) if pcm.size else 0.0
    db = lambda x: round(20 * np.log10(x), 1) if x > 0 else -200.0  # noqa: E731
    return {"duration_s": round(pcm.size / SR, 2), "peak_db": db(peak), "rms_db": db(rms)}


def grade_segment(seg: dict) -> tuple[str, list[str]]:
    """→ (tier, flags). tier ∈ confident | uncertain | rejected."""
    text = norm(seg.get("text", ""))
    nsp, lp, cr = seg.get("no_speech_prob", 0.0), seg.get("avg_logprob", 0.0), seg.get("compression_ratio", 0.0)
    if not text:
        return "rejected", ["no_text"]                 # empty, "♪", punctuation only
    if cr > COMPRESSION_RATIO_MAX:
        return "rejected", ["repetitive"]              # decoding loop
    if nsp > NO_SPEECH_PROB_MAX and lp < AVG_LOGPROB_MIN:
        return "rejected", ["no_speech"]               # Whisper's own silence rule
    flags = []
    if text in HALLUCINATION_PHRASES:
        flags.append("common_hallucination_phrase")
    if lp < AVG_LOGPROB_MIN:
        flags.append("low_logprob")
    if nsp > NO_SPEECH_PROB_MAX:
        flags.append("high_no_speech_prob")
    return ("uncertain" if flags else "confident"), flags


def transcribe_pcm(pcm: np.ndarray) -> dict:
    """Returns a transcript dict. Never raises: Whisper errors are reported as status=error."""
    lv = levels(pcm)
    base = {"model": config.WHISPER_MODEL, "audio": lv, "language": None, "text": "", "uncertain_text": "",
            "segments": [], "rejected_segments": [], "status": "ok", "error": None}
    if lv["peak_db"] < SILENT_PEAK_DB:
        return {**base, "audio_class": "silent", "status": "skipped"}
    try:
        import mlx_whisper

        with _whisper_lock:
            r = mlx_whisper.transcribe(pcm, path_or_hf_repo=config.WHISPER_MODEL,
                                       condition_on_previous_text=False, verbose=None)
    except Exception as e:  # noqa: BLE001
        return {**base, "audio_class": "unknown", "status": "error", "error": f"{type(e).__name__}: {e}"}

    segments, rejected = [], []
    for s in r.get("segments", []):
        seg = {"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": s["text"].strip(),
               "avg_logprob": round(float(s.get("avg_logprob", 0)), 3),
               "no_speech_prob": round(float(s.get("no_speech_prob", 0)), 3),
               "compression_ratio": round(float(s.get("compression_ratio", 0)), 3)}
        tier, flags = grade_segment(seg)
        seg.update(confidence="high" if tier == "confident" else "low", flags=flags)
        (rejected if tier == "rejected" else segments).append(seg)
    confident = [s for s in segments if s["confidence"] == "high"]
    return {**base, "audio_class": "speech" if confident else "non_speech",
            "language": r.get("language"), "segments": segments, "rejected_segments": rejected,
            "text": " ".join(s["text"] for s in confident),
            "uncertain_text": " ".join(s["text"] for s in segments if s["confidence"] == "low")}


def transcribe_file(path: Path) -> dict:
    try:
        pcm = decode_pcm(path)
    except AudioDecodeError as e:
        return {"model": config.WHISPER_MODEL, "audio": None, "audio_class": "missing", "language": None,
                "text": "", "uncertain_text": "", "segments": [], "rejected_segments": [],
                "status": "error", "error": f"decode: {e}"}
    return transcribe_pcm(pcm)
