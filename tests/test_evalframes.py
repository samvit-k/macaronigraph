"""Opt-in evaluation frames: disabled by default, selected Reels only, safe paths, purge."""
import json

import pytest
from fastapi.testclient import TestClient

from backend import config, evalframes
from backend.app import app
from conftest import PLAY_WALL, clip_upload

FRAMES = [(0.0, b"\xff\xd8jpeg0"), (1.0, b"\xff\xd8jpeg1")]


def jpgs():
    return list(evalframes.eval_dir().rglob("*.jpg")) if evalframes.eval_dir().exists() else []


def test_disabled_by_default_writes_nothing(isolated_output):
    assert evalframes.selection() == []
    assert evalframes.save("R1", "ig-a", FRAMES, 1.0, None) is None
    assert jpgs() == []


def test_only_selected_reels_are_saved(isolated_output):
    evalframes.enable(["R1"])
    assert evalframes.save("R2", "ig-a", FRAMES, 1.0, None) is None          # not selected
    d = evalframes.save("R1", "ig-a", FRAMES, 1.0, {"segments": [{"start": 0, "end": 1, "text": "hi"}], "text": "hi"})
    assert d is not None and len(jpgs()) == 2
    frames, meta = evalframes.load("R1")
    assert [t for t, _ in frames] == [0.0, 1.0] and meta["transcript_text"] == "hi"


def test_selection_is_capped(isolated_output):
    with pytest.raises(ValueError):
        evalframes.enable([f"R{i}" for i in range(evalframes.MAX_SELECTED + 1)])
    with pytest.raises(ValueError):
        evalframes.enable([])
    # a hand-edited selection longer than the cap is truncated
    evalframes.eval_dir().mkdir(parents=True)
    (evalframes.eval_dir() / "selection.json").write_text(json.dumps({"reel_ids": [f"R{i}" for i in range(9)]}))
    assert len(evalframes.selection()) == evalframes.MAX_SELECTED


@pytest.mark.parametrize("bad", ["../etc", "/abs", "a/b", "..", "x" * 65, "", "r\x00", "R 1", "R1;rm"])
def test_unsafe_ids_rejected(isolated_output, bad):
    with pytest.raises(evalframes.UnsafeIdError):
        evalframes.enable([bad])
    evalframes.enable(["R1"])
    with pytest.raises(evalframes.UnsafeIdError):
        evalframes.save("R1", bad, FRAMES, 1.0, None)                        # unsafe session id
    with pytest.raises(evalframes.UnsafeIdError):
        evalframes.load(bad)
    assert all(evalframes.eval_dir().resolve() in p.resolve().parents for p in jpgs())


def test_unsafe_ids_in_selection_file_are_ignored(isolated_output):
    evalframes.eval_dir().mkdir(parents=True)
    (evalframes.eval_dir() / "selection.json").write_text(json.dumps({"reel_ids": ["../../x", "OK1"]}))
    assert evalframes.selection() == ["OK1"]


def test_purge_deletes_everything_and_disables(isolated_output):
    evalframes.enable(["R1"])
    evalframes.save("R1", "ig-a", FRAMES, 1.0, None)
    assert evalframes.purge() == 2
    assert not evalframes.eval_dir().exists() and evalframes.selection() == []


def _process(client, rid, sid):
    audio, frames, walls = clip_upload(rid)
    meta = {"reel_id": rid, "source": "test", "session_id": sid, "playback": "ok", "video_duration_s": 12,
            "expected_s": 12, "captured_s": 12, "play_wall": PLAY_WALL, "rec_start_wall": PLAY_WALL,
            "frame_walls": walls, "frame_interval_ms": 1000}
    files = [("audio", ("a.webm", audio, "audio/webm"))] + [("frames", (f"f{i}.jpg", f, "image/jpeg"))
                                                             for i, f in enumerate(frames)]
    return client.post("/api/process", data={"meta": json.dumps(meta)}, files=files).json()


def test_normal_processing_never_stores_frames_but_selected_does(isolated_output):
    client = TestClient(app)
    _process(client, "TESTCLIP01", "test-normal")
    assert jpgs() == []                                                       # mode off: nothing stored
    evalframes.enable(["TESTCLIP02"])
    _process(client, "TESTCLIP01", "test-sel1")                               # not selected
    assert jpgs() == []
    _process(client, "TESTCLIP02", "test-sel2")                               # selected
    stored = jpgs()
    assert stored and all("TESTCLIP02" in p.parts for p in stored)
    media = [p for ext in ("*.mp4", "*.webm", "*.wav", "*.jpg") for p in config.OUTPUT_DIR.rglob(ext)]
    assert all(evalframes.eval_dir() in p.parents for p in media)            # no media outside eval dir
    assert not any(p.suffix in (".mp4", ".webm", ".wav") for p in media)     # frames only, never video/audio
