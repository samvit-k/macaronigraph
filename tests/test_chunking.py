"""Chunking: transcript windows, OCR screen grouping, speech-echo suppression, fragments and provenance."""
import copy
import json

from backend import chunking, record

def make_record(transcript=None, ocr_segs=None, caption="How to plan a weekly meal prep #cooking", duration=30.0,
                rid="R1", status="ok"):
    tr = {"segments": transcript} if transcript is not None else None
    oc = {"segments": ocr_segs or [], "frame_interval_s": 1.0}
    rec = {"reel_id": rid, "source_url": f"https://www.instagram.com/reel/{rid}/", "status": status,
           "saved_at": "2026-11-22T00:00:00+00:00", "pipeline_version": "phase2-0.1",
           "metadata": {"caption": caption, "hashtags": ["ai"], "owner": {"username": "someone"}},
           "capture": {"video_duration_s": duration}, "transcript": tr, "ocr": oc}
    rec["passages"] = record.passages(rid, rec["source_url"], caption, tr, oc)
    return rec

def seg(start, end, text, conf="high"):
    return {"start": start, "end": end, "text": text, "confidence": conf}

def oseg(start, end, text, kind="text", echo="on_screen", quality="normal", y=0.5, x=0.1):
    return {"start": start, "end": end, "text": text, "confidence": 90.0, "box": [x, y, x + 0.5, y + 0.03],
            "kind": kind, "echo": {"score": 0.0, "label": echo}, "quality": quality}

def by_type(chunks, t):
    return [c for c in chunks if c["metadata"]["content_type"] == t]

def ids_of(c):
    return json.loads(c["metadata"]["passage_ids"])

def test_transcript_windows_overlap_and_timestamps():
    sentence = "this sentence has exactly ten words in it for testing."
    tr = [seg(i * 4.0, i * 4.0 + 4.0, sentence) for i in range(14)]
    chunks = by_type(chunking.build_chunks(make_record(tr), "instagram"), "transcript")
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        assert ids_of(a)[-1] == ids_of(b)[0]
        assert a["metadata"]["start"] < b["metadata"]["start"]
    assert chunks[0]["metadata"]["start"] == 0.0
    assert chunks[-1]["metadata"]["end"] == 56.0
    assert all(c["metadata"]["n_words"] <= chunking.MAX_WORDS + 10 for c in chunks)

def test_short_transcript_is_one_chunk_and_low_conf_flagged():
    tr = [seg(0, 2, "Hello there."), seg(2, 4, "uh maybe", conf="low")]
    (c,) = by_type(chunking.build_chunks(make_record(tr), "instagram"), "transcript")
    assert c["text"] == "Hello there. uh maybe"
    assert c["metadata"]["has_low_conf"] is True

def test_list_screen_keeps_lines_in_reading_order_with_newlines():
    ocr = [oseg(5, 8, "3. Deploy", y=0.6), oseg(5, 8, "1. Install uv", y=0.2), oseg(5, 8, "2. Run pytest", y=0.4)]
    (c,) = by_type(chunking.build_chunks(make_record([], ocr), "instagram"), "ocr")
    assert c["text"] == "1. Install uv\n2. Run pytest\n3. Deploy"

def test_reading_order_survives_slightly_different_detection_times():
    ocr = [oseg(5.6, 8, "1. Install uv", y=0.2),
           oseg(5.3, 8, "2. Run pytest", y=0.4),
           oseg(5.45, 8, "3. Deploy", y=0.6),
           oseg(7.0, 9, "Next: monitoring", y=0.2)]
    (c,) = by_type(chunking.build_chunks(make_record([], ocr), "instagram"), "ocr")
    assert c["text"] == "1. Install uv\n2. Run pytest\n3. Deploy\nNext: monitoring"

def test_separate_moments_become_separate_screens():
    ocr = [oseg(1, 2, "First tip"), oseg(10, 11, "Second tip")]
    assert len(by_type(chunking.build_chunks(make_record([], ocr), "instagram"), "ocr")) == 2

def test_persistent_title_kept_even_if_spoken():
    tr = [seg(0, 4, "Former line cook shares how to prep vegetables fast")]
    ocr = [oseg(0, 28, "Former-line cook shares how to prep vegetables fast", echo="speech_echo")]
    chunks = chunking.build_chunks(make_record(tr, ocr), "instagram")
    (t,) = by_type(chunks, "ocr_title")
    assert "Former-line" in t["text"] and t["metadata"]["persistent"] is True

def test_confidently_spoken_subtitle_dropped_but_on_screen_text_kept():
    tr = [seg(0, 3, "home cooks keep repeating the word mise")]
    ocr = [oseg(0, 1, "home cooks keep", kind="subtitle", echo="speech_echo"),
           oseg(0, 1, "KNIFE vs PAN vs OVEN", echo="on_screen", y=0.1)]
    rec = make_record(tr, ocr)
    plan = chunking.ocr_plan(rec)
    assert [p["text"] for p in plan["dropped_echo"]] == ["home cooks keep"]
    (c,) = by_type(chunking.build_chunks(rec, "instagram"), "ocr")
    assert c["text"] == "KNIFE vs PAN vs OVEN"

def test_no_speech_reel_keeps_all_ocr():
    ocr = [oseg(0, 1, "home cooks keep", kind="subtitle", echo="speech_echo")]
    chunks = chunking.build_chunks(make_record(None, ocr, status="ok_no_speech"), "instagram")
    assert by_type(chunks, "ocr")[0]["text"] == "home cooks keep"

def test_echo_not_dropped_when_matching_speech_is_low_confidence():
    tr = [seg(0, 3, "intro words here"), seg(20, 23, "use the agents md file", conf="low")]
    ocr = [oseg(21, 22, "use the agents md file", kind="subtitle", echo="speech_echo")]
    plan = chunking.ocr_plan(make_record(tr, ocr))
    assert not plan["dropped_echo"] and len(plan["screen"]) == 1

def test_echo_not_dropped_when_confident_speech_is_elsewhere_in_the_reel():
    tr = [seg(0, 3, "use the agents md file")]
    ocr = [oseg(28, 29, "use the agents md file", kind="subtitle", echo="speech_echo")]
    plan = chunking.ocr_plan(make_record(tr, ocr, duration=40))
    assert not plan["dropped_echo"] and len(plan["screen"]) == 1

def test_echo_not_dropped_when_nearby_confident_speech_says_something_else():
    tr = [seg(5, 8, "completely unrelated narration")]
    ocr = [oseg(6, 7, "pip install chromadb", kind="subtitle", echo="speech_echo")]
    assert not chunking.ocr_plan(make_record(tr, ocr))["dropped_echo"]

def test_fragments_get_their_own_identifiable_chunks():
    ocr = [oseg(3, 4, "Install the CLI"), oseg(3, 4, "aN", quality="fragment", y=0.8)]
    chunks = chunking.build_chunks(make_record([], ocr), "instagram")
    (s,) = by_type(chunks, "ocr")
    (f,) = by_type(chunks, "ocr_fragment")
    assert s["text"] == "Install the CLI" and f["text"] == "aN"
    assert f["metadata"]["ocr_quality"] == "fragment" and "fragment" not in s["metadata"]["ocr_quality"]

def test_persistent_fragment_stays_a_fragment_not_a_title():
    ocr = [oseg(0, 28, "Clauud Codee", quality="fragment"), oseg(0, 28, "Real Title Here")]
    chunks = chunking.build_chunks(make_record([], ocr), "instagram")
    (t,) = by_type(chunks, "ocr_title")
    (f,) = by_type(chunks, "ocr_fragment")
    assert t["text"] == "Real Title Here" and t["metadata"]["ocr_quality"] == "normal"
    assert f["text"] == "Clauud Codee"
    assert f["metadata"]["ocr_quality"] == "fragment" and f["metadata"]["persistent"] is True
    ocr += [oseg(2, 3, "xQ", quality="fragment"), oseg(20, 21, "zz", quality="fragment")]
    frags = by_type(chunking.build_chunks(make_record([], ocr), "instagram"), "ocr_fragment")
    assert sorted(c["text"] for c in frags) == ["Clauud Codee", "xQ", "zz"]

def test_repeated_lines_keep_all_provenance():
    ocr = [oseg(1, 2, "Step 1: plan"), oseg(1, 2, "Step 2: build", y=0.6), oseg(2.5, 3.5, "Step 1: plan")]
    rec = make_record([], ocr)
    (c,) = by_type(chunking.build_chunks(rec, "instagram"), "ocr")
    assert c["text"] == "Step 1: plan\nStep 2: build"
    ocr_ids = [p["chunk_id"] for p in rec["passages"] if p["source_type"] == "ocr"]
    assert sorted(ids_of(c)) == sorted(ocr_ids)
    assert (c["metadata"]["start"], c["metadata"]["end"]) == (1, 3.5)

def test_split_screen_keeps_repeat_provenance_in_part_with_first_occurrence(monkeypatch):
    monkeypatch.setattr(chunking, "MAX_CHUNK_CHARS", 30)
    ocr = [oseg(1, 2, "alpha line one", y=0.1), oseg(1, 2, "beta line two", y=0.2),
           oseg(1, 2, "gamma line three", y=0.3), oseg(1.5, 2, "alpha line one", y=0.9)]
    rec = make_record([], ocr)
    parts = by_type(chunking.build_chunks(rec, "instagram"), "ocr")
    assert len(parts) >= 2
    all_ids = [i for c in parts for i in ids_of(c)]
    assert sorted(all_ids) == sorted(p["chunk_id"] for p in rec["passages"] if p["source_type"] == "ocr")
    assert sum(c["text"].count("alpha line one") for c in parts) == 1

def test_every_kept_passage_is_accounted_for():
    tr = [seg(0, 3, "home cooks keep repeating the word mise")]
    ocr = [oseg(0, 1, "home cooks keep", kind="subtitle", echo="speech_echo"),
           oseg(0, 28, "Title"), oseg(4, 5, "x", quality="fragment"), oseg(4, 5, "On screen", y=0.3),
           oseg(4.5, 5, "On screen", y=0.3)]
    rec = make_record(tr, ocr)
    chunks = chunking.build_chunks(rec, "instagram")
    plan = chunking.ocr_plan(rec)
    in_chunks = {i for c in chunks for i in ids_of(c) if ":ocr:" in i}
    dropped = {p["chunk_id"] for p in plan["dropped_echo"]}
    all_ocr = {p["chunk_id"] for p in rec["passages"] if p["source_type"] == "ocr"}
    assert in_chunks | dropped == all_ocr and not (in_chunks & dropped)

def _legacy(rec):
    for s in rec["ocr"]["segments"]:
        for k in ("echo", "quality", "kind"):
            s.pop(k)
    for p in rec["passages"]:
        if p["source_type"] == "ocr":
            p["ocr_echo"] = p["ocr_kind"] = p["ocr_quality"] = None
    return rec

def test_legacy_record_without_echo_labels_gets_them_in_memory():
    tr = [seg(0, 3, "home cooks keep repeating the word mise")]
    ocr = [oseg(0, 1, "home cooks keep"),
           oseg(0, 1, "Ready in twenty minutes", y=0.1)]
    rec = _legacy(make_record(tr, ocr))
    before = copy.deepcopy(rec)
    plan = chunking.ocr_plan(rec)
    assert rec == before
    (dropped,) = plan["dropped_echo"]
    assert dropped["text"] == "home cooks keep" and dropped["echo"] == "speech_echo"
    (kept,) = plan["screen"]
    assert kept["text"] == "Ready in twenty minutes" and kept["echo"] == "on_screen"

def test_legacy_record_without_transcript_keeps_everything():
    rec = _legacy(make_record(None, [oseg(0, 1, "home cooks keep")], status="ok_no_speech"))
    plan = chunking.ocr_plan(rec)
    assert not plan["dropped_echo"] and plan["screen"][0]["echo"] == "unknown"

def test_chunks_are_deterministic_and_hash_tracks_content():
    tr = [seg(0, 3, "hello world.")]
    ocr = [oseg(1, 2, "Install uv")]
    a = chunking.build_chunks(make_record(tr, ocr), "instagram")
    b = chunking.build_chunks(make_record(tr, ocr), "instagram")
    assert a == b and chunking.content_hash(a) == chunking.content_hash(b)
    c = chunking.build_chunks(make_record(tr, [oseg(1, 2, "Install pip")]), "instagram")
    assert chunking.content_hash(c) != chunking.content_hash(a)
    assert [x["id"] for x in a] == ["R1:caption:000", "R1:transcript:000", "R1:ocr:000"]
    for x in a:
        assert all(isinstance(v, (str, int, float, bool)) for v in x["metadata"].values())

def test_embed_text_has_context_header():
    (cap, tr) = chunking.build_chunks(make_record([seg(0, 2, "hi.")]), "instagram")[:2]
    assert chunking.embed_text(tr).startswith("Reel by @someone — How to plan a weekly meal prep · spoken\n")
    assert chunking.embed_text(cap).startswith("Reel by @someone · caption\n")