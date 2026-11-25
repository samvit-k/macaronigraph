"""Ground-truth metrics for the generated test media (used by pytest and scripts/validate.py)."""
from __future__ import annotations

from difflib import SequenceMatcher

from .transcribe import norm

def wer(reference: str, hypothesis: str) -> float:
    r, h = norm(reference).split(), norm(hypothesis).split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)

def card_found(card_text: str, segments: list[dict]) -> dict | None:
    want = norm(card_text)
    for s in segments:
        got = norm(s["text"])
        if want in got or SequenceMatcher(None, want, got).ratio() >= 0.85:
            return s
    return None

def ocr_card_recall(cards: list[dict], segments: list[dict], min_frames_visible: float = 0.0) -> float:
    eligible = [c for c in cards if c["end"] - c["start"] >= min_frames_visible]
    if not eligible:
        return 1.0
    return sum(card_found(c["text"], segments) is not None for c in eligible) / len(eligible)

def ocr_false_lines(cards: list[dict], segments: list[dict]) -> list[str]:
    return [s["text"] for s in segments if not any(card_found(c["text"], [s]) for c in cards)]