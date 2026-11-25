"""scripts/validate.py checks against generated ground truth."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate import validate  # noqa: E402

TRUTH = {
    "A": {"audio_class": "speech", "script": "hello world", "cards": [{"text": "HI THERE", "start": 0, "end": 3}]},
    "S": {"audio_class": "silent", "script": "", "cards": []},
    "M": {"audio_class": None, "expect_status": "failed", "script": "", "cards": []},
}

def rec(rid, status, audio_class, text="", ocr=("HI THERE",), sid="test-1"):
    return {"reel_id": rid, "session_id": sid, "status": status, "errors": [],
            "transcript": {"audio_class": audio_class, "text": text},
            "ocr": {"segments": [{"text": t, "start": 0.5, "end": 2.5} for t in ocr]}, "capture": {}}

GOOD_A = rec("A", "ok", "speech", "hello world")

def test_complete_subset_passes():
    _, fails = validate("test-1", [GOOD_A], TRUTH, ["A"])
    assert fails == []

def test_empty_session_fails():
    _, fails = validate("test-1", [], TRUTH, ["A"])
    assert any("no records" in f for f in fails) and any("A: expected" in f for f in fails)

def test_missing_expected_item_fails():
    _, fails = validate("test-1", [GOOD_A], TRUTH, ["A", "S"])
    assert fails == ["S: expected but no record in session"]

def test_no_manifest_fails():
    _, fails = validate("test-1", [GOOD_A], TRUTH, None)
    assert any("cannot verify completeness" in f for f in fails)

def test_audio_class_mismatch_fails_even_if_status_matches():
    r = rec("S", "ok_no_speech", "non_speech", ocr=())
    _, fails = validate("test-1", [r], TRUTH, ["S"])
    assert fails == ["S: audio_class non_speech != silent"]

def test_unexpected_or_unknown_reel_fails():
    _, fails = validate("test-1", [GOOD_A, rec("Z", "ok", "speech")], TRUTH, ["A"])
    assert any("Z: no ground truth" in f for f in fails)
    _, fails = validate("test-1", [GOOD_A, rec("S", "ok_no_speech", "silent", ocr=())], TRUTH, ["A"])
    assert any("S: processed but not in the expected list" in f for f in fails)

def test_other_sessions_ignored_and_failed_item_expected():
    rows = [rec("A", "ok", "speech", "hello world", sid="test-0"), rec("M", "failed", None, ocr=())]
    rows[1]["transcript"] = None
    rows[1]["ocr"] = None
    _, fails = validate("test-1", rows, TRUTH, ["M"])
    assert fails == []