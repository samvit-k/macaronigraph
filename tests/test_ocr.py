"""OCR clean-up: noise filtering, blocks, speech echo, subtitle merging, repeats and fragment labels."""
import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from backend import ocr

FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 56)

def card(text: str | None, bg="#1f3a5f") -> bytes:
    img = Image.new("RGB", (720, 1280), bg)
    if text:
        ImageDraw.Draw(img).text((60, 600), text, fill="white", font=FONT)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()

def test_dedupes_repeated_text_into_time_range():
    frames = [(t, card("HELLO WORLD")) for t in (0, 1, 2)] + [(3, card("SECOND CARD")), (4, card(None))]
    r = ocr.ocr_frames(frames)
    assert r["status"] == "ok" and r["frames_duplicate"] >= 2
    assert [(s["text"], s["start"], s["end"]) for s in r["segments"]] == [
        ("HELLO WORLD", 0, 2), ("SECOND CARD", 3, 3)]

def test_reappearing_text_after_short_gap_merges():
    frames = [(0, card("SAME TEXT")), (1, card(None)), (2, card("SAME TEXT"))]
    segs = ocr.ocr_frames(frames)["segments"]
    assert len(segs) == 1 and segs[0]["end"] == 2

def test_no_text_is_ok_with_zero_segments():
    r = ocr.ocr_frames([(0, card(None)), (1, card(None, bg="#000"))])
    assert r["status"] == "ok" and r["segments"] == []

@pytest.mark.parametrize("text,conf", [("RQ", 63.0), ("ee", 60.0), ("Ag", 65.0), ("a a a", 88.0), ("Bo \\", 67.5),
                                       ("@ #", 90.0), ("aN", 78.0)])
def test_noise_lines_from_real_instagram_frames_are_filtered(text, conf):
    assert ocr.noise_reason(ocr.Line(text, conf, (0, 0, 1, 1))) is not None

@pytest.mark.parametrize("text,conf", [("AI", 95.0), ("WSL", 84.5), ("x86", 70.0), ("ARM", 90.3), ("Q4", 92.0),
                                       ("Eat-Sleep-Code-Repeat", 64.0), ("eep reasoning", 89.0)])
def test_short_but_meaningful_text_is_kept(text, conf):
    assert ocr.noise_reason(ocr.Line(text, conf, (0, 0, 1, 1))) is None

def test_merge_text_collapses_rolling_overlap():
    assert ocr.merge_text("a sharp knife", "sharp knife cuts cleanly") == "a sharp knife cuts cleanly"
    assert ocr.merge_text("a sharp knife", "a sharp knife cuts") == "a sharp knife cuts"
    assert ocr.merge_text("the whole line", "whole") == "the whole line"
    assert ocr.merge_text("home cooks keep", "chopping all day") == "home cooks keep chopping all day"

def test_group_blocks_joins_multiline_caption_but_not_distant_title():
    title = ocr.Line("22yo chef explains", 90, (0.1, 0.05, 0.9, 0.09))
    l1 = ocr.Line("Can chop & simmer", 95, (0.2, 0.50, 0.8, 0.54))
    l2 = ocr.Line("Uses pans, ovens", 95, (0.2, 0.555, 0.8, 0.595))
    l3 = ocr.Line("& timers", 82, (0.4, 0.61, 0.6, 0.65))
    blocks = ocr.group_blocks([l3, title, l1, l2])
    assert [b.text for b in blocks] == ["22yo chef explains", "Can chop & simmer Uses pans, ovens & timers"]

def _raw(i, start, text, y, frames=1, end=None):
    return {"id": i, "start": start, "end": start if end is None else end, "text": text, "confidence": 95.0,
            "frames": frames, "box": [0.1, y, 0.9, y + 0.04]}

def test_merge_subtitles_chains_same_band_and_keeps_provenance():
    raw = [_raw(0, 0.0, "22yo chef explains", 0.05, frames=50, end=49.0),
           _raw(1, 1.0, "home cooks keep", 0.70),
           _raw(2, 2.0, "chopping all", 0.71),
           _raw(3, 3.0, "day long", 0.70),
           _raw(4, 3.0, "Saucepan", 0.30),
           _raw(5, 9.0, "much later caption", 0.70)]
    out = ocr.merge_subtitles(raw, interval_s=1.0)
    subs = [s for s in out if s["kind"] == "subtitle"]
    assert len(subs) == 1 and subs[0]["text"] == "home cooks keep chopping all day long"
    assert subs[0]["parts"] == [1, 2, 3] and (subs[0]["start"], subs[0]["end"]) == (1.0, 3.0)
    texts = {s["text"] for s in out if s["kind"] == "text"}
    assert texts == {"22yo chef explains", "Saucepan", "much later caption"}
    assert [s["id"] for s in out] == list(range(len(out)))

def _frame(*items) -> bytes:
    img = Image.new("RGB", (720, 1280), "#1f3a5f")
    d = ImageDraw.Draw(img)
    for text, y, size in items:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
        w = d.textlength(text, font=f)
        d.text(((720 - w) / 2, y), text, fill="white", font=f)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()

SPEECH = [{"start": 0.0, "end": 4.0, "text": "Caching stores the result of expensive work"}]

def test_pipeline_echo_gated_merge_never_absorbs_unrelated_text():
    frames = [(0.0, _frame(("CACHING STORES", 900, 56), ("MENU", 100, 40))),
              (1.0, _frame(("THE RESULT OF", 900, 56), ("MENU", 100, 40))),
              (2.0, _frame(("PROMO CODE SAVE", 900, 56), ("MENU", 100, 40))),
              (3.0, _frame(("EXPENSIVE WORK", 900, 56), ("MENU", 100, 40)))]
    r = ocr.ocr_frames(frames, interval_s=1.0, transcript_segments=SPEECH)
    assert r["speech_aligned"] is True
    subs = [s for s in r["segments"] if s["kind"] == "subtitle"]
    texts = [s["text"] for s in r["segments"]]
    assert any("PROMO CODE SAVE" == t for t in texts)
    assert all("PROMO" not in s["text"] for s in subs)
    promo = next(s for s in r["segments"] if s["text"] == "PROMO CODE SAVE")
    assert promo["echo"]["label"] == "on_screen" and promo["kind"] == "text"
    assert subs and subs[0]["text"].startswith("CACHING STORES THE RESULT OF")
    assert subs[0]["echo"]["label"] == "speech_echo" and "Caching" in subs[0]["aligned_speech"]
    raw_ids = {s["id"] for s in r["raw_segments"]}
    assert all(set(s["parts"]) <= raw_ids for s in r["segments"])
    assert {p for s in r["segments"] for p in s["parts"]} == raw_ids
    assert any(s["text"] == "MENU" and s["echo"]["label"] == "on_screen" for s in r["segments"])

def test_pipeline_without_transcript_labels_unknown_and_still_works():
    frames = [(0.0, _frame(("HELLO THERE", 600, 56))), (1.0, _frame(("HELLO THERE", 600, 56)))]
    r = ocr.ocr_frames(frames)
    assert r["speech_aligned"] is False and r["segments"][0]["echo"]["label"] == "unknown"

def test_geometry_blocks_merging_different_font_sizes_in_same_band():
    raw = [_raw(0, 1.0, "caching stores", 0.70), _raw(1, 2.0, "the result", 0.70)]
    raw[1]["box"] = [0.1, 0.69, 0.9, 0.79]
    out = ocr.merge_subtitles(raw, 1.0)
    assert [s["kind"] for s in out] == ["text", "text"]
    assert ocr.merge_subtitles(raw, 1.0, geometry=False)[0]["kind"] == "subtitle"

def test_consolidate_repeats_keeps_every_occurrence_and_part():
    segs = [{"id": 0, "start": 2.4, "end": 2.4, "text": "10.", "confidence": 94, "frames": 1, "box": [0] * 4,
             "kind": "text", "parts": [3]},
            {"id": 1, "start": 5.0, "end": 6.0, "text": "Other", "confidence": 90, "frames": 2, "box": [0] * 4,
             "kind": "text", "parts": [4]},
            {"id": 2, "start": 13.4, "end": 15.4, "text": "10.", "confidence": 96, "frames": 3, "box": [0] * 4,
             "kind": "text", "parts": [7]}]
    out = ocr.consolidate_repeats(segs)
    ten = next(s for s in out if s["text"] == "10.")
    assert len(out) == 2 and ten["occurrences"] == [[2.4, 2.4], [13.4, 15.4]]
    assert ten["parts"] == [3, 7] and ten["frames"] == 4 and ten["end"] == 15.4

def test_echo_score_is_fuzzy_but_not_absolute():
    speech = {"caching", "stores", "result"}
    assert ocr.echo_score("CACHlNG STORES", speech) == 1.0
    assert ocr.echo_score("PROMO CODE", speech) == 0.0
    assert ocr.echo_label(0.4) == "partial_echo"

def _qseg(text, conf=90.0, frames=1, kind="text", echo="on_screen", **kw):
    return {"text": text, "confidence": conf, "frames": frames, "kind": kind, "echo": {"label": echo}, **kw}

@pytest.mark.parametrize("text,conf", [
    ("package.json", 91), ("Polished CGI", 95), ("Pixel cartoon", 96), ("Claymation", 95), ("Est. revenue", 94),
    ("Pricing pag", 89), ("eep reasoning", 91), ("Plain", 60), ("Lockers", 62), ("AP", 90), ("5 files", 91),
    ("GPT-4", 70), ("x86", 70), ("src/app", 75)])
def test_meaningful_text_is_never_labelled_fragment(text, conf):
    segs = [_qseg(text, conf)]
    ocr.label_quality(segs, raw=[], transcript_segments=[], context_text="")
    assert segs[0]["quality"] == "normal", segs[0]

@pytest.mark.parametrize("text,conf", [("ompiex", 82), ("aliSim", 60), ("YOUINEED", 90), ("VWUIR UUL", 71.5),
                                       ("injthe", 63), ("hod", 63)])
def test_isolated_scraps_are_labelled_fragment_with_reasons(text, conf):
    segs = [_qseg(text, conf)]
    ocr.label_quality(segs, raw=[], transcript_segments=[], context_text="")
    assert segs[0]["quality"] == "fragment" and "single_frame" in segs[0]["quality_reasons"]
    assert segs[0]["text"] == text

def test_fragment_rules_respect_context_recurrence_frames_and_echo():
    cases = {
        "spoken": (_qseg("ompiex"), dict(transcript_segments=[{"start": 0, "end": 1, "text": "ompiex"}])),
        "caption": (_qseg("orbitfx"), dict(context_text="Try orbitfx today #orbitfx")),
        "recurring": (_qseg("vultr"), dict(raw=[{"text": "vultr gpu"}, {"text": "vultr"}])),
        "two_frames": (_qseg("ompiex", frames=2), {}),
        "repeated": (_qseg("ompiex", occurrences=[[1, 1], [5, 5]]), {}),
        "caption_kind": (_qseg("ompiex", kind="subtitle"), {}),
        "echo": (_qseg("ompiex", echo="partial_echo"), {}),
        "long": (_qseg("aa bb cc", conf=60), {}),
    }
    for name, (seg, kw) in cases.items():
        args = {"raw": [], "transcript_segments": [], "context_text": "", **kw}
        ocr.label_quality([seg], **args)
        assert seg["quality"] == "normal", name

def test_without_wordlist_labelling_gets_more_conservative(monkeypatch):
    monkeypatch.setattr(ocr, "_WORDS", set())
    segs = [_qseg("ompiex", 82), _qseg("hod", 63)]
    ocr.label_quality(segs, raw=[], transcript_segments=[], context_text="")
    assert [s["quality"] for s in segs] == ["normal", "fragment"]

def test_fragments_stay_in_passages_with_quality_flag():
    from backend.record import passages

    o = {"segments": [{"start": 1.0, "end": 1.0, "text": "ompiex", "confidence": 82, "kind": "text",
                       "parts": [0], "echo": {"label": "on_screen"}, "quality": "fragment"}]}
    (p,) = passages("R", "u", "", None, o)
    assert p["text"] == "ompiex" and p["ocr_quality"] == "fragment"

def test_filtered_noise_is_retained_not_discarded():
    img = Image.new("RGB", (720, 1280), "#1f3a5f")
    d = ImageDraw.Draw(img)
    d.text((60, 600), "REAL TEXT HERE", fill="white", font=FONT)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    r = ocr.ocr_frames([(0, buf.getvalue())])
    assert "REAL TEXT HERE" in r["text"]
    assert isinstance(r["filtered_segments"], list) and r["raw_segments"]

def test_corrupt_and_missing_frames():
    r = ocr.ocr_frames([(0, b"not a jpeg")])
    assert r["status"] == "error" and r["frame_errors"] == 1
    assert ocr.ocr_frames([])["status"] == "no_frames"