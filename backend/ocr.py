"""Frame OCR with Tesseract: in-memory only (frame bytes are never written to disk).

Pipeline per item
  1. per frame: Tesseract lines (words with conf ≥ MIN_WORD_CONF) + normalized bounding boxes
  2. noise filter: junk lines are moved to `filtered_segments` (with reason), never silently dropped
  3. per frame: vertically adjacent lines of similar size are grouped into one text block
  4. over time: identical/similar blocks are tracked into time ranges → `raw_segments`
  5. subtitles: consecutive short-lived segments in the same screen band (burned-in, word-by-word
     captions) are merged into one `kind="subtitle"` segment whose `parts` reference the raw segments
Output `segments` = persistent text + merged subtitles (what passages are built from).
"""
from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from PIL import Image, ImageOps

from . import config
from .transcribe import norm

MIN_WORD_CONF = 60
SIMILAR = 0.85              # block text similarity to treat as the same overlay
TESS_CONFIG = "--psm 11"    # sparse text: overlays scattered over the frame
# Duplicate frames: count changed pixels (a small text change on a static background barely moves the mean).
THUMB, PIXEL_CHANGE, DUP_CHANGED_FRAC = 128, 30, 0.0005
# Subtitle merging
TRANSIENT_MAX_S = 2.5       # visible ≤ this long → short-lived (independent of the sampling interval)
BAND_TOL = 0.06             # same subtitle band: |Δ y-center| ≤ 6 % of frame height
MAX_SUBTITLE_WORDS = 80     # cap a merged subtitle block (≈ a long sentence or two)
X_CENTER_TOL = 0.15         # same caption: horizontal centres within 15 % of frame width
# Speech alignment (labels only — never used to delete text)
ECHO_PAD_S = 2.0            # Whisper segment timing is coarse; compare against speech ± this window
ECHO_HIGH, ECHO_LOW = 0.6, 0.3
ECHO_MERGE = 0.5            # fragments merge into a caption only if at least this much is spoken
REPEAT_SIMILAR = 0.9        # repeated on-screen text consolidated into one segment with occurrences


@dataclass
class Line:
    text: str
    conf: float
    box: tuple  # (x0, y0, x1, y1) normalized to 0..1

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class _Seg:
    text: str
    start: float
    end: float
    box: tuple
    confs: list = field(default_factory=list)
    frames: int = 0

    def out(self, sid: int) -> dict:
        return {"id": sid, "start": round(self.start, 2), "end": round(self.end, 2), "text": self.text,
                "confidence": round(float(np.mean(self.confs)), 1), "frames": self.frames,
                "box": [round(v, 3) for v in self.box]}


# ---------------------------------------------------------------- per-frame OCR

def _prepare(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    if g.width < 1000:
        g = g.resize((g.width * 2, g.height * 2), Image.LANCZOS)
    if np.asarray(g).mean() < 128:     # light text on dark background → invert for Tesseract
        g = ImageOps.invert(g)
    return ImageOps.autocontrast(g)


def ocr_lines(img: Image.Image) -> list[Line]:
    """Tesseract lines built from words with conf ≥ MIN_WORD_CONF, with normalized boxes."""
    import pytesseract

    p = _prepare(img)
    d = pytesseract.image_to_data(p, config=TESS_CONFIG, output_type=pytesseract.Output.DICT)
    groups: dict[tuple, list[int]] = {}
    for i, word in enumerate(d["text"]):
        if word.strip() and float(d["conf"][i]) >= MIN_WORD_CONF:
            groups.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(i)
    lines = []
    for idx in groups.values():
        idx.sort(key=lambda i: d["left"][i])
        x0 = min(d["left"][i] for i in idx) / p.width
        y0 = min(d["top"][i] for i in idx) / p.height
        x1 = max(d["left"][i] + d["width"][i] for i in idx) / p.width
        y1 = max(d["top"][i] + d["height"][i] for i in idx) / p.height
        lines.append(Line(" ".join(d["text"][i] for i in idx),
                          float(np.mean([float(d["conf"][i]) for i in idx])), (x0, y0, x1, y1)))
    return lines


def noise_reason(line: Line) -> str | None:
    """Why a line is OCR junk (texture/edge artifacts), or None to keep it. Conservative: short but
    confident tokens ("AI", "x86", "WSL") are kept."""
    t = line.text.strip()
    words = re.findall(r"\w+", t)
    alnum = sum(ch.isalnum() for ch in t)
    if alnum < 2:
        return "too_short"
    if alnum / max(1, len(t.replace(" ", ""))) < 0.5:
        return "mostly_symbols"
    if len(words) >= 2 and len({w.lower() for w in words}) == 1 and len(words[0]) == 1:
        return "repeated_char"          # "a a a"
    if all(len(w) <= 2 for w in words) and line.conf < 85:
        return "short_tokens_low_conf"  # "RQ", "ee", "Ag"
    return None


def group_blocks(lines: list[Line]) -> list[Line]:
    """Merge vertically adjacent lines of similar height that overlap horizontally (one caption or
    paragraph split across lines) into a single block, read top-to-bottom."""
    blocks: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: (l.box[1], l.box[0])):
        for b in blocks:
            last = b[-1]
            gap = ln.box[1] - last.box[3]
            similar_size = 0.65 <= ln.h / max(last.h, 1e-6) <= 1.5
            overlap = min(ln.box[2], max(x.box[2] for x in b)) - max(ln.box[0], min(x.box[0] for x in b))
            if -0.2 * last.h <= gap <= 0.7 * min(ln.h, last.h) and similar_size and overlap > 0:
                b.append(ln)
                break
        else:
            blocks.append([ln])
    out = []
    for b in blocks:
        out.append(Line(" ".join(x.text for x in b), float(np.mean([x.conf for x in b])),
                        (min(x.box[0] for x in b), min(x.box[1] for x in b),
                         max(x.box[2] for x in b), max(x.box[3] for x in b))))
    return out


# ---------------------------------------------------------------- temporal processing

def _thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.grayscale(img).resize((THUMB, THUMB)), dtype=np.float32)


def _is_duplicate(a: np.ndarray, b: np.ndarray) -> bool:
    return float((np.abs(a - b) > PIXEL_CHANGE).mean()) < DUP_CHANGED_FRAC


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, norm(a), norm(b)).ratio() >= SIMILAR


def merge_text(a: str, b: str) -> str:
    """Join consecutive caption fragments, collapsing rolling/karaoke overlap
    ("a sharp knife" + "sharp knife cuts cleanly" → "a sharp knife cuts cleanly")."""
    na, nb = norm(a), norm(b)
    if not nb or nb in na:
        return a
    if na in nb:
        return b
    wa, wb = na.split(), nb.split()
    for k in range(min(len(wa), len(wb)), 0, -1):
        if wa[-k:] == wb[:k]:
            return f"{a} {' '.join(b.split()[k:])}".strip()
    return f"{a} {b}"


def _speech_words(transcript_segments: list[dict] | None, start: float, end: float, pad: float) -> set[str]:
    words: set[str] = set()
    for t in transcript_segments or []:
        if t["end"] >= start - pad and t["start"] <= end + pad:
            words.update(norm(t["text"]).split())
    return words


def echo_score(text: str, speech: set[str]) -> float:
    """Share of OCR tokens also spoken nearby (fuzzy for longer words, since both OCR and Whisper err)."""
    tokens = [w for w in norm(text).split() if len(w) >= 2 or w.isdigit()]
    if not tokens:
        return 0.0
    hit = sum(1 for w in tokens if w in speech or
              (len(w) >= 4 and any(SequenceMatcher(None, w, x).ratio() >= 0.8 for x in speech if len(x) >= 3)))
    return hit / len(tokens)


def echo_label(score: float | None) -> str:
    if score is None:
        return "unknown"          # no transcript to compare against
    return "speech_echo" if score >= ECHO_HIGH else "partial_echo" if score >= ECHO_LOW else "on_screen"


def annotate_echo(segs: list[dict], transcript_segments: list[dict] | None, interval_s: float) -> None:
    """Label (never delete) each segment by overlap with words spoken around the same time.
    A match is evidence of duplication, not proof: Whisper and OCR both make mistakes."""
    for s in segs:
        if transcript_segments is None:
            s["echo"] = {"score": None, "label": "unknown"}
            continue
        sc = echo_score(s["text"], _speech_words(transcript_segments, s["start"], s["end"], ECHO_PAD_S + interval_s))
        s["echo"] = {"score": round(sc, 2), "label": echo_label(sc)}


def _same_caption_geometry(a: list, b: list, size_range: tuple = (0.7, 1.4)) -> bool:
    """Same band, similar font size, similar horizontal centre."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    return (abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) <= BAND_TOL
            and size_range[0] <= hb / max(ha, 1e-6) <= size_range[1]
            and abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) <= X_CENTER_TOL)


# Caption tools auto-shrink long lines; when BOTH fragments are strongly confirmed by speech, tolerate that.
CONFIRMED_SIZE_RANGE = (0.5, 2.0)


def merge_subtitles(raw: list[dict], interval_s: float, transcript_segments: list[dict] | None = None,
                    geometry: bool = True) -> list[dict]:
    """Chain short-lived caption fragments into one `subtitle` segment (`parts` → raw segment ids).

    v2 (default): a fragment joins a chain only if its geometry matches the chain's latest fragment
    (band, font size, x-centre) and — when a transcript is given — only if it echoes the speech
    (echo score ≥ ECHO_MERGE). Short-lived text that does not echo speech is never merged: it is kept
    as its own segment. geometry=False with no transcript reproduces v1 (band-only) for comparison.
    Every output segment carries `parts`, so provenance back to raw_segments is uniform."""
    out: list[dict] = []
    open_chains: list[dict] = []
    max_gap = 1.5 * interval_s + 0.05
    use_echo = transcript_segments is not None
    if use_echo and any("echo" not in s for s in raw):
        annotate_echo(raw, transcript_segments, interval_s)
    for s in sorted(raw, key=lambda s: (s["start"], s["box"][1])):
        short_lived = s["end"] - s["start"] + interval_s <= TRANSIENT_MAX_S  # observed span + 1 sample
        echo_ok = not use_echo or (s["echo"]["score"] or 0) >= ECHO_MERGE
        if not short_lived or not echo_ok:
            out.append({**s, "kind": "text", "parts": [s["id"]]})
            continue
        yc = (s["box"][1] + s["box"][3]) / 2
        s_echo = (s.get("echo") or {}).get("score") or 0

        def fits(c):
            if not geometry:
                return abs(c["yc"] - yc) <= BAND_TOL
            confirmed = use_echo and s_echo >= ECHO_HIGH and c["last_echo"] >= ECHO_HIGH
            return _same_caption_geometry(c["last_box"], s["box"],
                                          CONFIRMED_SIZE_RANGE if confirmed else (0.7, 1.4))

        chain = next((c for c in open_chains if 0 <= s["start"] - c["end"] <= max_gap
                      and len(c["text"].split()) < MAX_SUBTITLE_WORDS and fits(c)), None)
        if chain is None:
            chain = {"yc": yc, "start": s["start"], "end": s["end"], "text": s["text"], "parts": [s["id"]],
                     "confs": [s["confidence"]], "frames": s["frames"], "box": list(s["box"]),
                     "last_box": list(s["box"]), "last_echo": s_echo}
            open_chains.append(chain)
        else:
            chain["text"] = merge_text(chain["text"], s["text"])
            chain["end"] = max(chain["end"], s["end"])
            chain["parts"].append(s["id"])
            chain["confs"].append(s["confidence"])
            chain["frames"] += s["frames"]
            chain["last_box"] = list(s["box"])
            chain["last_echo"] = s_echo
            chain["yc"] = yc
            chain["box"] = [min(chain["box"][0], s["box"][0]), min(chain["box"][1], s["box"][1]),
                            max(chain["box"][2], s["box"][2]), max(chain["box"][3], s["box"][3])]
    by_id = {s["id"]: s for s in raw}
    for c in open_chains:
        if len(c["parts"]) == 1:
            out.append({**by_id[c["parts"][0]], "kind": "text", "parts": c["parts"]})
            continue
        seg = {"start": c["start"], "end": c["end"], "text": c["text"],
               "confidence": round(float(np.mean(c["confs"])), 1), "frames": c["frames"],
               "box": c["box"], "kind": "subtitle", "parts": c["parts"]}
        if use_echo:
            scores = [by_id[p]["echo"]["score"] for p in c["parts"]]
            seg["echo"] = {"score": round(float(np.mean(scores)), 2), "label": echo_label(float(np.mean(scores)))}
            # the words Whisper heard over the same span: a readable companion to the raw OCR text
            seg["aligned_speech"] = " ".join(t["text"] for t in transcript_segments
                                             if t["end"] >= c["start"] - 0.5 and t["start"] <= c["end"] + interval_s)
        out.append(seg)
    out.sort(key=lambda s: (s["start"], s["box"][1]))
    for i, s in enumerate(out):
        s["id"] = i
    return out


def consolidate_repeats(segs: list[dict]) -> list[dict]:
    """Collapse non-adjacent repeats of the same on-screen text into one segment that keeps every
    occurrence (time range) and every raw part, so nothing is lost but duplication drops."""
    out: list[dict] = []
    for s in segs:
        if s["kind"] == "text":
            twin = next((o for o in out if o["kind"] == "text"
                         and SequenceMatcher(None, norm(o["text"]), norm(s["text"])).ratio() >= REPEAT_SIMILAR), None)
            if twin is not None:
                twin.setdefault("occurrences", [[twin["start"], twin["end"]]]).append([s["start"], s["end"]])
                twin["parts"] = twin["parts"] + s["parts"]
                twin["end"] = max(twin["end"], s["end"])
                twin["frames"] += s["frames"]
                if s["confidence"] > twin["confidence"]:
                    twin["text"], twin["confidence"] = s["text"], s["confidence"]
                continue
        out.append(dict(s))
    for i, s in enumerate(out):
        s["id"] = i
    return out


_WORDS: set[str] | None = None
FRAGMENT_MAX_WORDS = 2
FRAGMENT_LOW_CONF = 80


def _wordlist() -> set[str]:
    """System word list (lower-case). Empty if unavailable → no token is treated as a non-word scrap,
    which makes fragment labelling *more* conservative, never less."""
    global _WORDS
    if _WORDS is None:
        try:
            _WORDS = {w.strip().lower() for w in open("/usr/share/dict/words", encoding="utf-8", errors="ignore")
                      if w.strip()}
        except OSError:
            _WORDS = set()
    return _WORDS


def label_quality(segs: list[dict], raw: list[dict], transcript_segments: list[dict] | None,
                  context_text: str | None) -> None:
    """Add `quality` ("fragment" | "normal") + `quality_reasons` to output segments. Labels only: text,
    raw evidence and searchability are unchanged; Phase 3 can rank fragments lower.

    A segment is a fragment only if ALL hold: plain text (not caption/echo), seen in exactly one frame and
    never repeated, ≤ FRAGMENT_MAX_WORDS words, NO protected token, and (a non-word scrap OR low confidence).
    Protected tokens: contain a digit; recur in other raw segments; spoken; in the post caption/hashtags."""
    words = _wordlist()
    spoken = {w for t in transcript_segments or [] for w in norm(t["text"]).split()}
    context = set(norm(context_text or "").split())
    recurring = Counter(w for s in raw for w in set(norm(s["text"]).split()))

    def is_word(t: str) -> bool:  # dictionary word, allowing common inflections the old word list lacks
        if not words:
            return True  # no word list → never call anything a scrap
        stems = {t, re.sub(r"(ies)$", "y", t), re.sub(r"(es|s|ed|ing)$", "", t), re.sub(r"(ing|ed)$", "e", t)}
        return any(x in words for x in stems if len(x) >= 2)

    def protected_original(tok: str, conf: float) -> bool:  # judged on original casing/punctuation
        core = tok.strip("\"'“”‘’()[]{}.,;:!?")
        return bool(re.search(r"\w[./_]\w", core)                              # file names, paths: package.json
                    or (core[:1].isupper() and core[1:].islower() and len(core) >= 4)   # Names: Claymation
                    or (core.isupper() and core.isalpha() and 2 <= len(core) <= 5 and conf >= 85))  # acronyms

    for s in segs:
        toks = norm(s["text"]).split()
        reasons = []
        eligible = (s.get("kind") == "text" and (s.get("echo") or {}).get("label") in ("on_screen", "unknown")
                    and s.get("frames", 0) == 1 and not s.get("occurrences") and 0 < len(toks) <= FRAGMENT_MAX_WORDS)
        if eligible:
            protected = ([t for t in toks if any(c.isdigit() for c in t) or recurring[t] >= 2
                          or t in spoken or t in context or (len(t) >= 4 and is_word(t))]
                         + [o for o in s["text"].split() if protected_original(o, s["confidence"])])
            scraps = [t for t in toks if not is_word(t)]
            if not protected and (scraps or s["confidence"] < FRAGMENT_LOW_CONF):
                reasons = ["single_frame", f"≤{FRAGMENT_MAX_WORDS}_words", "no_protected_token"]
                reasons += ["non_word:" + ",".join(scraps)] if scraps else []
                reasons += [f"low_conf:{s['confidence']}"] if s["confidence"] < FRAGMENT_LOW_CONF else []
        s["quality"] = "fragment" if reasons else "normal"
        s["quality_reasons"] = reasons


def ocr_frames(frames: list[tuple[float, bytes]], interval_s: float = 1.0,
               transcript_segments: list[dict] | None = None, context_text: str | None = None) -> dict:
    """frames: [(timestamp_s, jpeg_bytes)] in any order; transcript_segments: Whisper segments in the same
    video time base (optional — enables speech-echo labels and echo-gated caption merging). Never raises."""
    config.configure_tesseract()
    frames = sorted(frames, key=lambda f: f[0])
    stats = {"frames_received": len(frames), "frames_ocr": 0, "frames_duplicate": 0,
             "frames_with_text": 0, "frame_errors": 0}
    segs: list[_Seg] = []
    active: list[_Seg] = []
    noise: dict[str, dict] = {}
    prev_thumb, prev_blocks, prev_noise = None, [], []
    error = None
    for t, data in frames:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception:  # noqa: BLE001
            stats["frame_errors"] += 1
            continue
        thumb = _thumb(img)
        if prev_thumb is not None and _is_duplicate(thumb, prev_thumb):
            stats["frames_duplicate"] += 1
            blocks, dropped = prev_blocks, prev_noise
        else:
            try:
                lines = ocr_lines(img)
            except Exception as e:  # noqa: BLE001
                stats["frame_errors"] += 1
                error = f"{type(e).__name__}: {e}"
                continue
            stats["frames_ocr"] += 1
            dropped = [(ln, r) for ln in lines if (r := noise_reason(ln))]
            blocks = group_blocks([ln for ln in lines if not noise_reason(ln)])
        prev_thumb, prev_blocks, prev_noise = thumb, blocks, dropped
        stats["frames_with_text"] += bool(blocks)
        for ln, reason in dropped:  # aggregated, kept for provenance/auditing
            n = noise.setdefault(norm(ln.text) or ln.text, {"text": ln.text, "start": t, "end": t, "frames": 0,
                                                             "confs": [], "reason": reason})
            n["end"], n["frames"] = t, n["frames"] + 1
            n["confs"].append(ln.conf)

        still_active = []
        for b in blocks:
            seg = next((s for s in active if _similar(s.text, b.text)), None)
            if seg is None:  # an identical overlay that reappears shortly after counts as the same one
                seg = next((s for s in reversed(segs) if t - s.end <= 2 * interval_s + 0.01
                            and _similar(s.text, b.text)), None)
            if seg is None:
                seg = _Seg(b.text, t, t, b.box)
                segs.append(seg)
            if b.conf > (max(seg.confs) if seg.confs else -1):
                seg.text = b.text  # keep the clearest reading
            seg.end, seg.frames = t, seg.frames + 1
            seg.confs.append(b.conf)
            still_active.append(seg)
        active = still_active

    status = "ok" if stats["frames_ocr"] or stats["frames_duplicate"] else ("error" if frames else "no_frames")
    if error and not stats["frames_ocr"]:
        status = "error"
    raw = [s.out(i) for i, s in enumerate(segs)]
    annotate_echo(raw, transcript_segments, interval_s)          # labels only; raw text untouched
    merged = consolidate_repeats(merge_subtitles(raw, interval_s, transcript_segments))
    label_quality(merged, raw, transcript_segments, context_text)  # labels only; nothing removed
    filtered = [{"text": n["text"], "start": round(n["start"], 2), "end": round(n["end"], 2), "frames": n["frames"],
                 "confidence": round(float(np.mean(n["confs"])), 1), "reason": n["reason"]} for n in noise.values()]
    return {"engine": f"tesseract {_tess_version()}", "config": TESS_CONFIG, "min_word_conf": MIN_WORD_CONF,
            "frame_interval_s": interval_s, "speech_aligned": transcript_segments is not None,
            "status": status, "error": error, **stats,
            "segments": merged, "raw_segments": raw, "filtered_segments": filtered,
            "text": "\n".join(s["text"] for s in merged)}


def _tess_version() -> str:
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception:  # noqa: BLE001
        return "unknown"
