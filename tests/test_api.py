"""Backend end-to-end: simulated extension uploads of the generated clips → JSONL, validated vs ground truth."""
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from backend import config, jobs, worker
from backend.app import app
from conftest import PLAY_WALL, clip_upload, ground_truth
from backend.validation import ocr_card_recall, wer

client = TestClient(app)


def post_clip(rid: str, **meta_extra):
    audio, frames, walls = clip_upload(rid)
    gt = ground_truth()[rid]
    meta = {"reel_id": rid, "source": "test", "session_id": "pytest", "playback": "ok",
            "video_duration_s": gt["duration"], "expected_s": gt["duration"], "captured_s": gt["duration"],
            "play_wall": PLAY_WALL, "rec_start_wall": PLAY_WALL, "frame_walls": walls,
            "frame_interval_ms": 1000, **meta_extra}
    files = [("audio", ("audio.webm", audio, "audio/webm"))] + \
            [("frames", (f"f{i}.jpg", f, "image/jpeg")) for i, f in enumerate(frames)]
    r = client.post("/api/process", data={"meta": json.dumps(meta)}, files=files)
    assert r.status_code == 200, r.text
    return r.json()


def assert_no_spooled_media():
    assert worker.pending_jobs() == [] and not list(worker.spool_dir().glob(".incoming-*"))


def records():
    """Attempt history (one line per processing attempt)."""
    return jobs.read_jsonl(jobs.attempts_jsonl())


@pytest.mark.parametrize("rid", ["TESTCLIP01", "TESTCLIP02", "TESTCLIP03"])
def test_speech_clip_ok_and_matches_ground_truth(isolated_output, rid):
    res = post_clip(rid)
    assert res["status"] == "ok", res
    rec = records()[-1]
    gt = ground_truth()[rid]
    assert wer(gt["script"], rec["transcript"]["text"]) < 0.15
    assert ocr_card_recall(gt["cards"], rec["ocr"]["segments"], min_frames_visible=1.0) == 1.0
    assert "🔍" in rec["metadata"]["caption"] or rid != "TESTCLIP01"
    assert rec["metadata"]["raw"]["label_values"][0]["label"] == "URL"
    assert rec["source_url"].endswith(f"/test-media/page/{rid}")
    # OCR segments are timestamped within each card's window (±1 sample interval)
    for c in gt["cards"]:
        if c["end"] - c["start"] >= 1.0:
            seg = next(s for s in rec["ocr"]["segments"] if c["text"] in s["text"])
            assert c["start"] - 1.0 <= seg["start"] <= c["end"] and seg["end"] <= c["end"] + 1.0
    assert_no_spooled_media()  # temp media deleted


def test_burned_in_subtitles_merge_into_one_segment_with_provenance(isolated_output):
    assert post_clip("TESTSUBS")["status"] == "ok"
    rec = records()[-1]
    gt = ground_truth()["TESTSUBS"]
    subs = [s for s in rec["ocr"]["segments"] if s["kind"] == "subtitle"]
    assert len(subs) == 1
    assert wer(" ".join(s["text"] for s in gt["subtitles"]), subs[0]["text"]) < 0.1
    raw_ids = {s["id"] for s in rec["ocr"]["raw_segments"]}
    assert set(subs[0]["parts"]) <= raw_ids and len(subs[0]["parts"]) >= 8
    ocr_passages = [p for p in rec["passages"] if p["source_type"] == "ocr"]
    assert {p["ocr_kind"] for p in ocr_passages} == {"text", "subtitle"} and len(ocr_passages) == 2
    assert wer(gt["script"], rec["transcript"]["text"]) < 0.15


def test_silent_and_music_clips_ok_no_speech(isolated_output):
    assert post_clip("TESTSILENT")["status"] == "ok_no_speech"
    assert post_clip("TESTMUSIC")["status"] == "ok_no_speech"
    for rec in records():
        assert rec["transcript"]["text"] == ""
        assert rec["ocr"]["segments"]


def test_unavailable_item_failed(isolated_output):
    meta = {"reel_id": "TESTMISSING", "source": "test", "playback": "unavailable", "playback_detail": "Page not found"}
    r = client.post("/api/process", data={"meta": json.dumps(meta)})
    assert r.json()["status"] == "failed"
    rec = records()[-1]
    assert rec["transcript"] is None and rec["ocr"] is None and rec["errors"] == ["unavailable: Page not found"]


def test_empty_upload_is_not_success(isolated_output):
    meta = {"reel_id": "TESTCLIP01", "source": "test", "playback": "ok", "expected_s": 12, "captured_s": 12,
            "video_duration_s": 12}
    r = client.post("/api/process", data={"meta": json.dumps(meta)},
                    files=[("audio", ("audio.webm", b"garbage", "audio/webm"))])
    assert r.json()["status"] == "failed"
    assert_no_spooled_media()


def test_session_queue_and_404_page(isolated_output):
    s = client.get("/api/session", params={"source": "test", "limit": 3}).json()
    assert [i["reel_id"] for i in s["items"]] == ["TESTCLIP01", "TESTCLIP02", "TESTCLIP03"]
    assert client.get("/test-media/page/TESTMISSING").status_code == 404
    assert client.get("/test-media/file/../../backend/app.py").status_code == 404


def test_source_json_unchanged(isolated_output):
    before = hashlib.sha256(config.TEST_POSTS_PATH.read_bytes()).hexdigest()
    post_clip("TESTCLIP02")
    assert hashlib.sha256(config.TEST_POSTS_PATH.read_bytes()).hexdigest() == before
