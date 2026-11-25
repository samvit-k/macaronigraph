"""Offline OCR comparison (v1 band-merge vs v2 speech-aligned) on opt-in evaluation frames.

Usage: .venv/bin/python scripts/ocr_eval.py [--show N] [--side-by-side REEL_ID]
Reads output/eval_frames/<reel>/<session>/ (see backend/evalframes.py). Tesseract runs once per Reel;
v1 and v2 are built from the same raw segments, so differences come only from merging/labelling.

These are PROXY metrics: real Reels have no ground truth. Ground-truth accuracy (WER, card recall) is
measured only on the generated test media (scripts/validate.py).

  segments     output segments (fewer = less fragmentation, if retention holds)
  readable     share of alphabetic tokens (≥3 letters) found in the system word list or the transcript
  garbled      share of multi-word segments whose readable share < 50 %
  retention    share of *informative* raw OCR tokens present in the output (see informative_tokens)
  screen-only  retention restricted to informative tokens never spoken (independent on-screen info)
  duplicate    share of segments whose normalized text repeats another segment's
Exit codes: 0 all selected Reels evaluated · 1 nothing evaluated · 2 incomplete (some Reels missing).
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import evalframes, ocr  # noqa: E402
from backend.transcribe import norm  # noqa: E402

WORDLIST = Path("/usr/share/dict/words")


def load_dict(path: Path = WORDLIST) -> set[str] | None:
    """System word list, or None if unavailable (readability is then reported as n/a)."""
    try:
        return {w.strip().lower() for w in path.read_text(encoding="utf-8", errors="ignore").splitlines() if w.strip()}
    except OSError:
        return None


def tokens(text: str) -> list[str]:
    return norm(text).split()


def informative_tokens(raw_segments: list[dict], words: set[str] | None, spoken: set[str]) -> set[str]:
    """Normalized tokens from raw OCR that plausibly carry information — not just dictionary words:
    numbers / alphanumerics (x86, 4, 13), ALL-CAPS acronyms (AI, RAG, WSL), dictionary or spoken words,
    and any ≥3-char term seen in ≥2 raw segments (recurring names like 'Vultr', 'Hacklytics')."""
    seen = Counter(t for s in raw_segments for t in set(tokens(s["text"])))
    out: set[str] = set()
    for s in raw_segments:
        for orig in re.findall(r"[\w][\w.+#-]*", s["text"]):
            for t in tokens(orig):
                if (any(ch.isdigit() for ch in t)
                        or (orig.isupper() and 2 <= len(orig) <= 6 and orig.isalpha())
                        or (len(t) >= 3 and ((words is not None and t in words) or t in spoken))
                        or (len(t) >= 3 and seen[t] >= 2)):
                    out.add(t)
    return out


def metrics(segs: list[dict], informative: set[str], words: set[str] | None, spoken: set[str]) -> dict:
    alpha = ok = garbled = multi = 0
    out_tokens: set[str] = set()
    for s in segs:
        toks = tokens(s["text"])
        out_tokens.update(toks)
        if words is None:
            continue
        a = [t for t in toks if t.isalpha() and len(t) >= 3]
        r = sum(t in words or t in spoken for t in a)
        alpha, ok = alpha + len(a), ok + r
        if len(a) >= 2:
            multi += 1
            garbled += r / len(a) < 0.5
    norms = Counter(norm(s["text"]) for s in segs)
    screen_only = {t for t in informative if t not in spoken}
    return {
        "segments": len(segs),
        "readable": (ok / alpha if alpha else 1.0) if words is not None else None,
        "garbled": (garbled / multi if multi else 0.0) if words is not None else None,
        "retention": len(informative & out_tokens) / len(informative) if informative else 1.0,
        "screen_only": len(screen_only & out_tokens) / len(screen_only) if screen_only else 1.0,
        "duplicate": sum(c - 1 for c in norms.values()) / len(segs) if segs else 0.0,
    }


def _context(reel_id: str) -> str:
    """Caption + hashtags from the source export (protected tokens for fragment labelling)."""
    try:
        from backend import config, metadata

        for p in metadata.load_posts(config.INSTAGRAM_POSTS_PATH):
            if p["reel_id"] == reel_id:
                return " ".join([p["metadata"]["caption"], *p["metadata"]["hashtags"]])
    except Exception:  # noqa: BLE001
        pass
    return ""


def fragment_stats(segs: list[dict], informative: set[str]) -> dict:
    frags = [s for s in segs if s.get("quality") == "fragment"]
    normal_tokens = {t for s in segs if s.get("quality") != "fragment" for t in tokens(s["text"])}
    at_risk = sorted({t for s in frags for t in tokens(s["text"]) if t in informative and t not in normal_tokens})
    return {"fragments": len(frags), "share": len(frags) / len(segs) if segs else 0.0, "at_risk": at_risk,
            "list": [(s["start"], s["confidence"], s["text"], s["quality_reasons"]) for s in frags]}


def evaluate_reel(frames, meta, words, reel_id=""):
    tsegs, interval = meta["transcript_segments"], meta["interval_s"]
    spoken = set(tokens(meta["transcript_text"]))
    r2 = ocr.ocr_frames(frames, interval_s=interval, transcript_segments=tsegs,
                        context_text=_context(reel_id) if reel_id else None)
    raw = r2["raw_segments"]
    v1 = ocr.merge_subtitles([{k: v for k, v in s.items() if k != "echo"} for s in raw], interval,
                             None, geometry=False)
    info = informative_tokens(raw, words, spoken)
    return {"frames": len(frames), "raw": raw, "filtered": r2["filtered_segments"], "v2": r2["segments"], "v1": v1,
            "m1": metrics(v1, info, words, spoken), "m2": metrics(r2["segments"], info, words, spoken),
            "echo": Counter(s["echo"]["label"] for s in r2["segments"]),
            "frag": fragment_stats(r2["segments"], info)}


def evaluate(selection: list[str], loader=evalframes.load, words: set[str] | None = None):
    """→ (results by reel, missing reel ids)."""
    results, missing = {}, []
    for rid in selection:
        try:
            frames, meta = loader(rid)
        except FileNotFoundError:
            missing.append(rid)
            continue
        if not frames:
            missing.append(rid)
            continue
        results[rid] = evaluate_reel(frames, meta, words, rid)
    return results, missing


def _pct(v):
    return "  n/a" if v is None else f"{v:.0%}"


def report(results: dict, missing: list[str], selection: list[str], words, show: int, side: str | None,
           list_fragments: bool = False) -> int:
    if words is None:
        print(f"WARNING: {WORDLIST} unavailable; readability/garbled reported as n/a")
    if missing:
        print(f"INCOMPLETE: no saved frames for {missing} ({len(results)}/{len(selection)} Reels evaluated)")
    if not results:
        print("NOTHING EVALUATED: capture the eval set first (popup Queue → 'OCR eval set (debug)')")
        return 1
    for rid, r in results.items():
        if show:
            print(f"\n--- {rid}: v2 sample ---")
            for s in r["v2"][:show]:
                extra = f" | speech: {s['aligned_speech'][:60]!r}" if s.get("aligned_speech") else ""
                occ = f" ×{len(s['occurrences'])}" if s.get("occurrences") else ""
                print(f"  {s['start']:5.1f}-{s['end']:5.1f} {s['kind']:8} {s['echo']['label']:12} "
                      f"{s['text'][:70]!r}{occ}{extra}")
        if side == rid:
            raw_by_id = {s["id"]: s for s in r["raw"]}
            print(f"\n--- {rid}: cleaned segment ⟵ raw fragments ---")
            for s in r["v2"]:
                print(f"  [{s['kind']}/{s['echo']['label']}] {s['text']!r}")
                for p in s["parts"]:
                    rs = raw_by_id[p]
                    print(f"      ⟵ {rs['start']:5.1f}-{rs['end']:5.1f} {rs['text']!r}")
            print(f"  filtered (kept in record): {[f['text'] for f in r['filtered']]}")

    print("\nPROXY METRICS: no ground truth exists for real Reels (see scripts/validate.py for test-media accuracy)")
    hdr = (f"{'reel':12} {'frames':>6} {'raw':>4} {'filt':>4} | {'ver':3} {'segs':>5} {'readable':>8} "
           f"{'garbled':>7} {'retain':>6} {'screen':>6} {'dup':>5} | v2 echo/partial/screen")
    print(hdr + "\n" + "-" * len(hdr))
    sums = {"m1": Counter(), "m2": Counter()}
    for rid, r in results.items():
        for v in ("m1", "m2"):
            m = r[v]
            lead = f"{rid:12} {r['frames']:>6} {len(r['raw']):>4} {len(r['filtered']):>4}" if v == "m1" else " " * 31
            tail = (f" | {r['echo']['speech_echo']}/{r['echo']['partial_echo']}/{r['echo']['on_screen']}"
                    if v == "m2" else "")
            print(f"{lead} | {'v' + v[1]:3} {m['segments']:>5} {_pct(m['readable']):>8} {_pct(m['garbled']):>7} "
                  f"{m['retention']:>6.0%} {m['screen_only']:>6.0%} {m['duplicate']:>5.0%}{tail}")
            sums[v].update({k: x for k, x in m.items() if x is not None})
    n = len(results)
    print("-" * len(hdr))
    for v in ("m1", "m2"):
        t = sums[v]
        rd = None if words is None else t["readable"] / n
        gb = None if words is None else t["garbled"] / n
        print(f"{'mean of ' + str(n):12} {'':>6} {'':>4} {'':>4} | {'v' + v[1]:3} {t['segments'] / n:>5.0f} "
              f"{_pct(rd):>8} {_pct(gb):>7} {t['retention'] / n:>6.0%} {t['screen_only'] / n:>6.0%} "
              f"{t['duplicate'] / n:>5.0%}")

    print("\nFRAGMENT LABELS (v2; labels only — every fragment is still in the record and searchable)")
    for rid, r in results.items():
        f = r.get("frag")
        if f is None:
            continue
        print(f"  {rid:12} {f['fragments']:>4} of {len(r['v2']):>4} segments ({f['share']:.0%}) labelled fragment; "
              f"informative tokens ONLY in fragments: {f['at_risk'] or 'none'}")
        if list_fragments:
            for start, conf, text, reasons in f["list"]:
                print(f"      {start:6.1f}s c={conf:<5} {text!r:28} {', '.join(reasons[3:])}")
    return 2 if missing else 0


def main(argv: list[str]) -> int:
    show = int(argv[argv.index("--show") + 1]) if "--show" in argv else 0
    side = argv[argv.index("--side-by-side") + 1] if "--side-by-side" in argv else None
    selection = evalframes.selection()
    if not selection:
        print("NOTHING EVALUATED: no eval selection enabled (python -m backend.evalframes enable …)")
        return 1
    words = load_dict()
    results, missing = evaluate(selection, words=words)
    return report(results, missing, selection, words, show, side, "--fragments" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
