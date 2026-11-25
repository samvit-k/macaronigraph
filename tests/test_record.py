"""Status rules: no false success."""
import pytest

from backend.record import build_record, decide_status

CAP_OK = {"playback": "ok", "expected_s": 12.0, "captured_s": 12.0, "video_duration_s": 12.0}
T_SPEECH = {"status": "ok", "audio_class": "speech", "text": "hello world", "uncertain_text": "", "segments": []}
T_SILENT = {"status": "skipped", "audio_class": "silent", "text": "", "uncertain_text": "", "segments": []}
T_MUSIC = {"status": "ok", "audio_class": "non_speech", "text": "", "uncertain_text": "Thank you.", "segments": []}
T_ERR = {"status": "error", "audio_class": "unknown", "error": "boom", "text": "", "uncertain_text": ""}
O_TEXT = {"status": "ok", "segments": [{"start": 0, "end": 1, "text": "HI", "confidence": 90}]}
O_EMPTY = {"status": "ok", "segments": []}

def test_fully_successful():
    assert decide_status(CAP_OK, T_SPEECH, O_TEXT) == ("ok", [])

def test_ocr_ok_with_zero_text_is_still_success():
    assert decide_status(CAP_OK, T_SPEECH, O_EMPTY) == ("ok", [])

@pytest.mark.parametrize("t", [T_SILENT, T_MUSIC])
def test_no_speech_success(t):
    assert decide_status(CAP_OK, t, O_TEXT) == ("ok_no_speech", [])

def test_missing_ocr_is_never_success():
    s, e = decide_status(CAP_OK, T_SPEECH, None)
    assert s == "partial" and any("ocr_missing" in x for x in e)
    s, e = decide_status(CAP_OK, T_SILENT, None)
    assert s == "failed" and any("ocr_missing" in x for x in e)

def test_missing_transcript_is_never_success():
    s, e = decide_status(CAP_OK, None, O_TEXT)
    assert s == "partial" and any("transcription_missing" in x for x in e)
    s, e = decide_status(CAP_OK, None, O_EMPTY)
    assert s == "failed"

def test_transcription_error():
    s, e = decide_status(CAP_OK, T_ERR, O_TEXT)
    assert s == "partial" and any("transcription_error: boom" in x for x in e)

def test_ocr_error_and_no_frames():
    for o in ({"status": "error", "error": "tess", "segments": []}, {"status": "no_frames", "segments": []}):
        s, e = decide_status(CAP_OK, T_SPEECH, o)
        assert s == "partial" and e

def test_incomplete_capture():
    s, e = decide_status({**CAP_OK, "captured_s": 6.0}, T_SPEECH, O_TEXT)
    assert s == "partial" and any("incomplete_capture" in x for x in e)

def test_truncated_long_video_cannot_succeed_even_with_full_coverage():
    cap = {"playback": "ok", "video_duration_s": 240.0, "expected_s": 180.0, "captured_s": 180.0,
           "truncated": True, "capture_limit_s": 180}
    for t in (T_SPEECH, T_SILENT):
        s, e = decide_status(cap, t, O_TEXT)
        assert s == "partial" and any("truncated_at_capture_limit" in x and "240.0" in x for x in e)
    s, _ = decide_status({**cap, "truncated": False}, T_SPEECH, O_TEXT)
    assert s == "partial"

def test_brief_mute_is_a_warning_not_an_error():
    cap = {**CAP_OK, "muted": True, "muted_s": 0.8}
    assert decide_status(cap, T_SPEECH, O_TEXT) == ("ok", [])
    rec = build_record({"reel_id": "R", "source_url": "u", "metadata": {}}, {**cap, "reel_id": "R"}, T_SPEECH, O_TEXT)
    assert rec["status"] == "ok" and rec["warnings"] == ["briefly_muted: 0.8s of 12.0s"]

def test_long_mute_is_an_error_and_legacy_mute_stays_strict():
    long_mute = {**CAP_OK, "muted": True, "muted_s": 6.0}
    s, e = decide_status(long_mute, T_SPEECH, O_TEXT)
    assert s == "partial" and any("audio_muted" in x and "6.0s" in x for x in e)
    assert decide_status({**CAP_OK, "muted": True}, T_SPEECH, O_TEXT)[0] == "partial"

def test_muted_playback_is_not_reported_as_no_speech():
    s, e = decide_status({**CAP_OK, "muted": True}, T_SILENT, O_TEXT)
    assert s == "partial" and any("audio_muted" in x for x in e)

@pytest.mark.parametrize("pb", ["unavailable", "no_video", "auth_required", "playback_blocked", "capture_lost"])
def test_fatal_playback_fails(pb):
    s, e = decide_status({"playback": pb, "playback_detail": "x"}, None, None)
    assert s == "failed" and e == [f"{pb}: x"]

def test_stalled_is_partial():
    s, e = decide_status({**CAP_OK, "playback": "stalled", "captured_s": 5}, T_SPEECH, O_TEXT)
    assert s == "partial" and "playback_stalled" in e

def test_record_passages_have_provenance():
    post = {"reel_id": "R1", "source_url": "https://www.instagram.com/reel/R1/", "saved_at": "x", "source_index": 0,
            "metadata": {"caption": "cap"}}
    t = {**T_SPEECH, "segments": [{"start": 0.0, "end": 1.0, "text": "hello world", "confidence": "high"}]}
    rec = build_record(post, {**CAP_OK, "reel_id": "R1"}, t, O_TEXT)
    ids = [p["chunk_id"] for p in rec["passages"]]
    assert ids == ["R1:caption:000", "R1:transcript:000", "R1:ocr:000"]
    assert all(p["url"] == post["source_url"] for p in rec["passages"])
    assert rec["status"] == "ok"

def test_unknown_metadata_downgrades():
    rec = build_record(None, {**CAP_OK, "reel_id": "ZZ", "url": "u"}, T_SPEECH, O_TEXT)
    assert rec["status"] == "partial" and any("metadata_not_found" in e for e in rec["errors"])