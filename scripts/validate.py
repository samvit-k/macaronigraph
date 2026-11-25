"""Validate a browser capture session against the generated media's ground truth.

Usage: .venv/bin/python scripts/validate.py [SESSION_ID] [--expect ID1,ID2,...]
  SESSION_ID defaults to the latest test-* session.
  Expected items come from the session manifest written by /api/session (so a 3-video run expects
  exactly those 3); --expect overrides it. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config, jobs  # noqa: E402
from backend.validation import card_found, ocr_card_recall, ocr_false_lines, wer  # noqa: E402

EXPECT = {"speech": "ok", "silent": "ok_no_speech", "non_speech": "ok_no_speech", None: "failed"}
MAX_WER = 0.15
MAX_SUBTITLE_WER = 0.2

def validate(sid: str, rows: list[dict], truth: dict, expected_ids: list[str] | None) -> tuple[list[str], list[str]]:
    """→ (report lines, failures). Pure: no file IO."""
    out, fails = [], []
    recs_all = [r for r in rows if r.get("session_id") == sid]
    latest: dict[str, dict] = {}
    for r in recs_all:
        latest[r["reel_id"]] = r  # last attempt wins
    out.append(f"session {sid}: {len(recs_all)} records, expected {len(expected_ids) if expected_ids is not None else '?'}\n")

    if expected_ids is None:
        fails.append("no session manifest found and no --expect given: cannot verify completeness")
    elif not expected_ids:
        fails.append("expected item list is empty")
    if not recs_all:
        fails.append(f"session {sid} has no records")
    if len(recs_all) > len(latest):
        out.append(f"note: {len(recs_all) - len(latest)} duplicate attempt(s); validating the latest per reel")

    hdr = f"{'reel':12} {'status':13} {'expect':13} {'audio':10} {'WER':>5} {'OCR':>5} {'flash':>5} {'Δt':>5} {'frames':>7} {'ivl ms':>6} live"
    out += [hdr, "-" * len(hdr)]
    for rid, r in latest.items():
        if rid not in truth:
            fails.append(f"{rid}: no ground truth for this reel")
            continue
        if expected_ids is not None and rid not in expected_ids:
            fails.append(f"{rid}: processed but not in the expected list")
        gt = truth[rid]
        exp = gt.get("expect_status") or EXPECT[gt.get("audio_class")]
        t, o, c = r.get("transcript") or {}, r.get("ocr") or {}, r.get("capture") or {}
        w = wer(gt["script"], t.get("text", "")) if gt.get("audio_class") == "speech" else None
        long_cards = [x for x in gt.get("cards", []) if x["end"] - x["start"] >= 1.0]
        flash = [x for x in gt.get("cards", []) if x["end"] - x["start"] < 1.0]
        recall = ocr_card_recall(long_cards, o.get("segments", [])) if o else None
        flash_hit = (f"{sum(card_found(x['text'], o.get('segments', [])) is not None for x in flash)}/{len(flash)}"
                     if flash and o else "-")
        # timestamp error: |first OCR sighting - card start| (bounded below by the sampling interval)
        dts = [abs(s["start"] - x["start"]) for x in long_cards if (s := card_found(x["text"], o.get("segments", [])))] if o else []
        dt = max(dts) if dts else None
        out.append(f"{rid:12} {r['status']:13} {exp:13} {str(t.get('audio_class', '-')):10} "
                   f"{'-' if w is None else f'{w:.2f}':>5} {'-' if recall is None else f'{recall:.2f}':>5} {flash_hit:>5} "
                   f"{'-' if dt is None else f'{dt:.1f}':>5} "
                   f"{str(c.get('frames_sampled', '-')) + '/' + str(c.get('frames_attempted', '-')):>7} "
                   f"{str(c.get('frame_interval_actual_ms', '-')):>6} {c.get('stream_live', '-')}")
        if r["status"] != exp:
            fails.append(f"{rid}: status {r['status']} != {exp} ({'; '.join(r.get('errors', []))})")
        if gt.get("audio_class") is not None and t.get("audio_class") != gt["audio_class"]:
            fails.append(f"{rid}: audio_class {t.get('audio_class')} != {gt['audio_class']}")
        if w is not None and w > MAX_WER:
            fails.append(f"{rid}: WER {w:.2f} > {MAX_WER}  hyp={t.get('text')!r}")
        if recall is not None and recall < 1.0:
            fails.append(f"{rid}: OCR recall {recall:.2f} on cards ≥1 s")
        if gt.get("subtitles") and o:
            subs = [s for s in o.get("segments", []) if s.get("kind") == "subtitle"]
            sw = wer(" ".join(s["text"] for s in gt["subtitles"]), " ".join(s["text"] for s in subs))
            out.append(f"{'':12} subtitles: {len(o.get('raw_segments', []))} fragments → {len(subs)} merged, WER {sw:.2f}")
            if sw > MAX_SUBTITLE_WER:
                fails.append(f"{rid}: subtitle WER {sw:.2f} > {MAX_SUBTITLE_WER}")
        if o and (noise := ocr_false_lines(gt.get("cards", []) + gt.get("subtitles", []),
                                           [s for s in o.get("segments", []) if s.get("kind") != "subtitle"])):
            out.append(f"{'':12} note: OCR lines not in ground truth: {noise}")
        if c.get("frames_dropped_timeout") or c.get("frames_encode_failed"):
            fails.append(f"{rid}: frames dropped={c.get('frames_dropped_timeout')} failed={c.get('frames_encode_failed')}")

    for rid in expected_ids or []:
        if rid not in latest:
            fails.append(f"{rid}: expected but no record in session")
    return out, fails

def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--expect")]
    expect_arg = next((a.split("=", 1)[1] if "=" in a else argv[argv.index(a) + 1]
                       for a in argv if a.startswith("--expect")), None)
    if expect_arg and expect_arg in args:
        args.remove(expect_arg)
    history = config.OUTPUT_DIR / "attempts.jsonl"  # every attempt (the canonical file keeps only the best)
    rows = jobs.read_jsonl(history) if history.is_file() else []
    sessions = [r["session_id"] for r in rows if (r.get("session_id") or "").startswith("test-")]
    sid = args[0] if args else (sessions[-1] if sessions else None)
    if not sid:
        print("FAILURES:\n  no test sessions in", history)
        return 1
    if expect_arg:
        expected = [x for x in expect_arg.split(",") if x]
    else:
        manifest = config.OUTPUT_DIR / "sessions" / f"{sid}.json"
        expected = [i["reel_id"] for i in json.loads(manifest.read_text())["items"]] if manifest.is_file() else None
    truth = json.loads(config.TEST_GROUND_TRUTH.read_text())
    report, fails = validate(sid, rows, truth, expected)
    spool = config.OUTPUT_DIR / "spool"
    leftovers = [p for p in spool.iterdir() if p.name != ".dead"] if spool.exists() else []
    if leftovers:
        fails.append(f"spooled media not yet processed/removed (wait for the worker): {leftovers}")
    print("\n".join(report))
    print("\n" + ("ALL CHECKS PASSED" if not fails else "FAILURES:\n  " + "\n  ".join(fails)))
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))