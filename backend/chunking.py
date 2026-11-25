"""Phase 3 chunking: canonical record → retrieval chunks (pure functions; the record is never modified).

Chunk kinds (`content_type`):
  caption       the Instagram caption
  transcript    consecutive Whisper segments merged into ~50–120-word windows, one-segment overlap
  ocr_title     long-lived overlays (titles, hooks) — kept once instead of repeated in every screen
  ocr           a "screen": on-screen text visible around the same time, lines in reading order
  ocr_fragment  OCR the pipeline labelled `fragment` (uncertain) — persistent or not. Always searchable,
                always identifiable (`ocr_quality=fragment`), down-weighted at ranking.

Speech duplication is handled conservatively: a subtitle / short-lived OCR passage is left out only if
CONFIDENT transcript segments around the same moment cover its words (echo score recomputed against those
segments alone). Persistent overlays are never left out; Reels without reliable speech keep all OCR.

Provenance: every source passage that contributed to a chunk — including exact repeats whose text is shown
only once — is listed in the chunk's `passage_ids`, and the chunk's time range spans all of them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from difflib import SequenceMatcher

from . import ocr as ocr_mod

CHUNKER_VERSION = "c1"
TARGET_WORDS = 50          # close a transcript window at the first sentence end after this many words
MAX_WORDS = 120            # …or unconditionally here
MIN_TAIL_WORDS = 15        # a smaller final window is folded into the previous one
SCREEN_GAP_S = 1.5         # OCR passages this close in time belong to the same screen
MAX_CHUNK_CHARS = 1200     # screens larger than this are split on line boundaries
TITLE_MIN_S = 10.0         # an overlay visible ≥ max(this, TITLE_FRACTION × duration) is persistent
TITLE_FRACTION = 0.4
DUP_SCREEN = 0.9           # near-identical screens are collapsed (provenance of both kept)
CAPTION_SHORT = 120
_SENT_END = re.compile(r"[.!?…][\"')\]]*$")

def _words(text: str) -> int:
    return len(text.split())

def _line_key(text: str) -> str:
    return " ".join(text.lower().split())

def _span(items: list[dict]) -> tuple[float | None, float | None]:
    st = [p["start"] for p in items if p.get("start") is not None]
    en = [p["end"] for p in items if p.get("end") is not None]
    return (round(min(st), 2) if st else None, round(max(en), 2) if en else None)

# ---------------------------------------------------------------- transcript

def transcript_windows(segs: list[dict]) -> list[list[dict]]:
    """Group transcript passages into windows; consecutive windows share one segment of overlap."""
    windows: list[list[dict]] = []
    i, n = 0, len(segs)
    while i < n:
        win, words = [], 0
        j = i
        while j < n:
            win.append(segs[j])
            words += _words(segs[j]["text"])
            j += 1
            if words >= MAX_WORDS or (words >= TARGET_WORDS and _SENT_END.search(segs[j - 1]["text"].strip())):
                break
        windows.append(win)
        if j >= n:
            break
        i = j - 1 if len(win) > 1 else j          # overlap: next window repeats this window's last segment
    if len(windows) > 1:
        tail_new = windows[-1][1:] if windows[-1][0] is windows[-2][-1] else windows[-1]
        if sum(_words(p["text"]) for p in tail_new) < MIN_TAIL_WORDS:
            windows[-2].extend(tail_new)
            windows.pop()
    return windows

# ---------------------------------------------------------------- OCR

def _ocr_items(record: dict) -> list[dict]:
    """OCR passages joined with their segment (for `box`), echo labels filled in for legacy records."""
    passages = [p for p in record.get("passages") or [] if p.get("source_type") == "ocr"]
    ocr = record.get("ocr") or {}
    segs = copy.deepcopy(ocr.get("segments") or [])
    if segs and len(segs) == len(passages) and any("echo" not in s for s in segs):
        # Phase-1 record: no echo labels yet → compute them in memory (same function the pipeline uses).
        tsegs = (record.get("transcript") or {}).get("segments")
        ocr_mod.annotate_echo([s for s in segs if "echo" not in s], tsegs, ocr.get("frame_interval_s", 1.0))
    items = []
    for i, p in enumerate(passages):
        seg = segs[i] if i < len(segs) else {}
        items.append({**p, "box": seg.get("box"),
                      "echo": p.get("ocr_echo") or (seg.get("echo") or {}).get("label") or "unknown",
                      "quality": p.get("ocr_quality") or "normal",
                      "kind": p.get("ocr_kind") or seg.get("kind") or "text"})
    return items

def _duration(record: dict) -> float:
    cap = record.get("capture") or {}
    for v in (cap.get("video_duration_s"), cap.get("captured_s"),
              ((record.get("transcript") or {}).get("audio") or {}).get("duration_s")):
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return 0.0

def _visible(p: dict) -> float:
    return (p["end"] or 0) - (p["start"] or 0) if p.get("start") is not None else 0.0

MOMENT_TOL_S = 1.0         # lines first detected within this of each other appeared together

def _order_key(p: dict):
    box = p.get("box") or [0, 0, 0, 0]
    return (round(p.get("start") or 0), round(box[1], 2), box[0])

def _reading_order(items: list[dict]) -> list[dict]:
    """Moments in time order; within a moment (detection times differing by ≤ MOMENT_TOL_S, e.g. because
    lines were first read on adjacent frames) top-to-bottom, then left-to-right."""
    moments: list[list[dict]] = []
    for p in sorted(items, key=lambda x: (x.get("start") or 0)):
        if moments and (p.get("start") or 0) - (moments[-1][0].get("start") or 0) <= MOMENT_TOL_S:
            moments[-1].append(p)
        else:
            moments.append([p])
    yx = lambda p: ((p.get("box") or [0, 0])[1], (p.get("box") or [0])[0])  # noqa: E731
    return [p for m in moments for p in sorted(m, key=yx)]

def confident_echo(p: dict, confident_segs: list[dict], interval_s: float) -> bool:
    """True only if confident speech around this passage's time covers its words (score ≥ ECHO_HIGH).
    Low-confidence or missing transcript coverage → False (the OCR is kept)."""
    if p.get("start") is None or not confident_segs:
        return False
    speech = ocr_mod._speech_words(confident_segs, p["start"], p["end"] if p.get("end") is not None else p["start"],
                                   ocr_mod.ECHO_PAD_S + interval_s)
    return bool(speech) and ocr_mod.echo_score(p["text"], speech) >= ocr_mod.ECHO_HIGH

def ocr_plan(record: dict) -> dict:
    """Decide what happens to every OCR passage → lists of passages under
    "title" (persistent, normal quality), "screen" (normal), "fragment" (uncertain, transient),
    "fragment_persistent" (uncertain, long-lived), "dropped_echo" (confidently spoken)."""
    items = _ocr_items(record)
    confident = [s for s in (record.get("transcript") or {}).get("segments") or [] if s.get("confidence") == "high"]
    interval = (record.get("ocr") or {}).get("frame_interval_s") or 1.0
    title_min = max(TITLE_MIN_S, TITLE_FRACTION * _duration(record))
    plan = {"title": [], "screen": [], "fragment": [], "fragment_persistent": [], "dropped_echo": []}
    for p in items:
        if not p["text"].strip():
            continue
        persistent = _visible(p) >= title_min
        transient = p["kind"] == "subtitle" or _visible(p) <= ocr_mod.TRANSIENT_MAX_S
        if p["quality"] == "fragment":                    # uncertain OCR stays identifiable everywhere
            plan["fragment_persistent" if persistent else "fragment"].append(p)
        elif persistent:
            plan["title"].append(p)
        elif transient and p["echo"] == "speech_echo" and confident_echo(p, confident, interval):
            plan["dropped_echo"].append(p)
        else:
            plan["screen"].append(p)
    return plan

def _group_by_time(items: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    end = None
    for p in sorted(items, key=lambda x: ((x.get("start") or 0), (x.get("end") or 0))):
        s, e = p.get("start") or 0, p.get("end") or 0
        if groups and end is not None and s <= end + SCREEN_GAP_S:
            groups[-1].append(p)
            end = max(end, e)
        else:
            groups.append([p])
            end = e
    return groups

def _split_lines(group: list[dict]) -> list[list[dict]]:
    """Order a screen's lines for reading and split at MAX_CHUNK_CHARS on line boundaries. A repeated line
    counts once toward the size, but its passage stays in the part holding the first occurrence."""
    parts: list[list[dict]] = [[]]
    home: dict[str, int] = {}
    size = 0
    for p in _reading_order(group):
        key = _line_key(p["text"])
        if key in home:
            parts[home[key]].append(p)                    # provenance kept; text shown once
            continue
        if parts[-1] and size + len(p["text"]) + 1 > MAX_CHUNK_CHARS:
            parts.append([])
            size = 0
        parts[-1].append(p)
        home[key] = len(parts) - 1
        size += len(p["text"]) + 1
    return [pt for pt in parts if pt]

def _screen_text(sc: list[dict]) -> str:
    lines, seen = [], set()
    for p in _reading_order(sc):
        key = _line_key(p["text"])
        if key not in seen:
            seen.add(key)
            lines.append(p["text"].strip())
    return "\n".join(lines)

def _collapse_duplicates(screens: list[list[dict]]) -> list[list[dict]]:
    out: list[tuple[str, list[dict]]] = []
    for sc in screens:
        text = _screen_text(sc)
        for prev_text, prev in out:
            if SequenceMatcher(None, prev_text, text).ratio() >= DUP_SCREEN:
                prev.extend(sc)      # every source passage kept; time range widens
                break
        else:
            out.append((text, list(sc)))
    return [sc for _, sc in out]

# ---------------------------------------------------------------- record → chunks

def _context(record: dict) -> tuple[str, str]:
    md = record.get("metadata") or {}
    owner = (md.get("owner") or {}).get("username") or ""
    caption = " ".join((md.get("caption") or "").split())
    first = caption.split("#")[0].strip() or caption
    return owner, first[:CAPTION_SHORT]

LABEL = {"caption": "caption", "transcript": "spoken", "ocr": "on-screen text",
         "ocr_title": "on-screen title", "ocr_fragment": "on-screen text (uncertain OCR)"}

def embed_text(chunk: dict) -> str:
    """What gets embedded: a short context header + the chunk text (the stored document is the plain text)."""
    md = chunk["metadata"]
    head = f"Reel by @{md['owner_username']}" if md.get("owner_username") else "Reel"
    if md.get("caption_short") and md["content_type"] != "caption":
        head += f" — {md['caption_short']}"
    return f"{head} · {LABEL[md['content_type']]}\n{chunk['text']}"

def build_chunks(record: dict, source: str) -> list[dict]:
    """Deterministic chunks for one canonical record: [{id, text, metadata}]; metadata values are scalars."""
    rid = record["reel_id"]
    owner, caption_short = _context(record)
    md = record.get("metadata") or {}
    base = {"reel_id": rid, "source": source, "source_url": record.get("source_url") or "",
            "owner_username": owner, "caption_short": caption_short, "saved_at": record.get("saved_at") or "",
            "record_status": record.get("status") or "", "pipeline_version": record.get("pipeline_version") or "",
            "chunker_version": CHUNKER_VERSION}
    chunks: list[dict] = []
    seq: dict[str, int] = {}

    def add(kind: str, text: str, items: list[dict], **extra):
        text = text.strip()
        if not text:
            return
        n = seq.get(kind, 0)
        seq[kind] = n + 1
        start, end = _span(items)
        meta = {**base, "content_type": kind, "passage_ids": json.dumps([p["chunk_id"] for p in items]),
                "n_words": _words(text), **extra}
        if start is not None:
            meta["start"] = start
        if end is not None:
            meta["end"] = end
        chunks.append({"id": f"{rid}:{kind}:{n:03d}", "text": text, "metadata": meta})

    passages = record.get("passages") or []
    for p in passages:
        if p.get("source_type") == "caption":
            add("caption", p["text"], [p], hashtags=" ".join(f"#{h}" for h in md.get("hashtags") or []))

    tsegs = [p for p in passages if p.get("source_type") == "transcript" and p.get("text", "").strip()]
    for win in transcript_windows(tsegs):
        add("transcript", " ".join(p["text"].strip() for p in win), win,
            has_low_conf=any(p.get("confidence") == "low" for p in win))

    plan = ocr_plan(record)

    def labels(items, persistent=False):
        return {"ocr_kinds": ",".join(sorted({p["kind"] for p in items})),
                "ocr_echo": ",".join(sorted({p["echo"] for p in items})),
                "ocr_quality": ",".join(sorted({p["quality"] for p in items})), "persistent": persistent}

    def singles(items):  # persistent passages: one chunk each (identical repeats collapsed)
        return _collapse_duplicates([[p] for p in sorted(items, key=_order_key)])

    def screens(items):
        return _collapse_duplicates([part for g in _group_by_time(items) for part in _split_lines(g)])

    for sc in singles(plan["title"]):
        add("ocr_title", _screen_text(sc), sc, **labels(sc, persistent=True))
    for sc in screens(plan["screen"]):
        add("ocr", _screen_text(sc), sc, **labels(sc))
    for sc in singles(plan["fragment_persistent"]):
        add("ocr_fragment", _screen_text(sc), sc, **labels(sc, persistent=True))
    for sc in screens(plan["fragment"]):
        add("ocr_fragment", _screen_text(sc), sc, **labels(sc))
    return chunks

def content_hash(chunks: list[dict]) -> str:
    """Hash of everything that ends up in the index for a Reel (ids, documents, embedded text, metadata)."""
    h = hashlib.sha256(CHUNKER_VERSION.encode())
    for c in chunks:
        h.update(json.dumps([c["id"], c["text"], embed_text(c), c["metadata"]], sort_keys=True,
                            ensure_ascii=False).encode())
    return h.hexdigest()[:24]

def summarize(record: dict, chunks: list[dict]) -> dict:
    """Counts used for reporting (how many OCR passages went where, chunks per type)."""
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["metadata"]["content_type"]] = by_type.get(c["metadata"]["content_type"], 0) + 1
    return {"chunks": by_type, "ocr_passages": {k: len(v) for k, v in ocr_plan(record).items()}}