"""Async processing: spool durability, DB failures, quarantine, crash recovery, races, session completion,
backpressure. Every test runs inside `isolated_output` (temp OUTPUT_DIR); mocks are restored individually."""
import io
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import config, jobs, worker
from backend.app import _lookup, app
from conftest import PLAY_WALL, clip_upload

LOOKUP = lambda source, rid: None  # noqa: E731
REAL_RECORD_ATTEMPT = jobs.record_attempt
REAL_RMTREE = worker.shutil.rmtree
REAL_MARK_QUEUED = jobs.mark_queued


@pytest.fixture(autouse=True)
def _iso(isolated_output, monkeypatch):
    jobs._export_state.update(pending=0, last=0.0)
    monkeypatch.setattr(worker, "RETRY_BACKOFF_S", 0)
    jobs.import_posts("ig", [{"reel_id": f"R{i}", "source_url": f"u/R{i}", "source_index": i, "media_type": "reel",
                              "saved_at": None} for i in range(4)])
    yield isolated_output
    # every artefact must live under the isolated temp OUTPUT_DIR
    assert isolated_output in Path(jobs.db_path()).parents and isolated_output in worker.spool_dir().parents


def lease(n=1, sid="ig-a"):
    return jobs.create_session(sid, "ig", n)


def cap(rid="R0", sid="ig-a", playback="ok", **kw):
    return {"reel_id": rid, "session_id": sid, "source": "ig", "playback": playback, "expected_s": 5,
            "captured_s": 5, "video_duration_s": 5, "frame_walls": [], **kw}


def spool_one(rid="R0", sid="ig-a", playback="unavailable"):
    return worker.spool(cap(rid, sid, playback, playback_detail="Page not found"), io.BytesIO(b"fake-audio"),
                        [io.BytesIO(b"\xff\xd8jpeg")])


def state(rid):
    with jobs.connect() as con:
        return dict(con.execute("SELECT state, lease_session, attempts FROM reels WHERE reel_id=?", (rid,)).fetchone())


def n_attempts(rid=None):
    with jobs.connect() as con:
        q = "SELECT COUNT(*) FROM attempts" + (" WHERE reel_id=?" if rid else "")
        return con.execute(q, (rid,) if rid else ()).fetchone()[0]


def db_down(*_a, **_k):
    raise sqlite3.OperationalError("database is locked")


GOOD_RECORD = {"reel_id": "R0", "session_id": "ig-a", "status": "failed", "errors": ["unavailable"],
               "processed_at": "2026-11-23T00:00:00+00:00", "capture": {}, "transcript": None, "ocr": None}


# ---------------------------------------------------------------- spool + basic processing

def test_spool_marks_queued_and_processing_removes_media():
    lease()
    job = spool_one()
    assert state("R0")["state"] == "queued" and (job / "audio.webm").exists()
    assert worker.run_once(LOOKUP)
    assert not job.exists() and n_attempts("R0") == 1 and state("R0")["state"] != "queued"


def test_identity_comes_from_meta_not_directory_name():
    lease()
    job = spool_one()
    renamed = job.with_name("999_R9__ig-zzz")             # misleading directory name
    job.rename(renamed)
    assert worker.job_identity(renamed) == ("ig-a", "R0")
    assert worker.find_job("ig-a", "R0") == renamed


# ---------------------------------------------------------------- DB failures

def test_db_failure_keeps_job_and_media_then_persists_without_reprocessing(monkeypatch):
    lease()
    job = spool_one(playback="ok")                       # fake audio → real (fast) decode-failure path
    builds, real_build = [], worker.build
    monkeypatch.setattr(worker, "build", lambda j, l: builds.append(1) or real_build(j, l))
    monkeypatch.setattr(jobs, "record_attempt", db_down)
    assert worker.run_once(LOOKUP)
    assert job.exists() and (job / "audio.webm").exists() and worker.valid_checkpoint(job)
    assert json.loads((job / "failures.json").read_text())["count"] == 1 and n_attempts() == 0
    monkeypatch.setattr(jobs, "record_attempt", REAL_RECORD_ATTEMPT)   # DB back
    assert worker.run_once(LOOKUP)
    assert not job.exists() and n_attempts("R0") == 1 and builds == [1]   # extraction ran once only


def test_repeated_persist_failures_quarantine_reconcile_and_drop_media(monkeypatch):
    lease()
    job = spool_one()
    monkeypatch.setattr(jobs, "record_attempt", db_down)
    for _ in range(worker.MAX_PERSIST_FAILURES):
        worker.run_once(LOOKUP)
    dead = worker.dead_dir() / job.name
    assert not job.exists() and dead.exists()
    assert worker.valid_checkpoint(dead) and (dead / "QUARANTINED.txt").exists()
    assert not (dead / "audio.webm").exists() and not (dead / "frames").exists()  # valid record → media dropped
    assert state("R0")["state"] != "queued"                                   # reconciled immediately


def test_invalid_metadata_quarantined_keeps_only_copy_and_reconciles():
    lease()
    jobs.mark_queued("R0", "ig-a")
    bad = worker.spool_dir() / "123_bad"
    (bad / "frames").mkdir(parents=True)
    (bad / "audio.webm").write_bytes(b"only copy")
    (bad / "meta.json").write_text("{not json")
    assert worker.run_once(LOOKUP)
    dead = worker.dead_dir() / "123_bad"
    assert dead.exists() and (dead / "audio.webm").read_bytes() == b"only copy"   # no record → media kept
    assert state("R0")["state"] == "pending" and state("R0")["lease_session"] is None  # reconciled → re-capturable
    assert worker.purge_dead(older_than_days=365) == 0 and (dead / "audio.webm").exists()  # explicit, aged only
    assert worker.purge_dead(older_than_days=0) == 2 and not (dead / "audio.webm").exists()


def test_corrupt_checkpoint_keeps_media_and_retry_rebuilds_from_it():
    lease(2)
    a, b = spool_one("R0"), spool_one("R1")
    (a / "record.json").write_text("{corrupt")
    assert worker.run_once(LOOKUP) and worker.run_once(LOOKUP)            # worker survives, continues
    dead = worker.dead_dir() / a.name
    assert dead.exists() and not b.exists() and n_attempts("R1") == 1
    assert (dead / "audio.webm").read_bytes() == b"fake-audio"            # corrupt checkpoint ≠ evidence
    assert (dead / "frames").exists() and (dead / "record.json.corrupt").exists()
    assert not (dead / "record.json").exists()                            # never reused
    assert state("R0")["state"] != "queued"
    assert worker.retry_dead() == 1 and worker.drain(LOOKUP) == 1         # rebuilt from the original media
    assert n_attempts("R0") == 1 and worker.pending_jobs() == [] and not dead.exists()


@pytest.mark.parametrize("bad", [
    {"reel_id": "R0"},                                              # missing keys
    [1, 2],                                                         # not a dict
    {**GOOD_RECORD, "reel_id": 123},                                # wrong type
    {**GOOD_RECORD, "reel_id": "../x"},                             # unsafe id
    {**GOOD_RECORD, "session_id": ""},                              # empty
    {**GOOD_RECORD, "status": "great"},                             # unknown status
    {**GOOD_RECORD, "errors": "unavailable"},                       # str instead of list
    {**GOOD_RECORD, "errors": [1, 2]},                              # list of non-str
    {**GOOD_RECORD, "processed_at": 1700000000},                    # number instead of str
    {**GOOD_RECORD, "capture": None},                               # capture must be a dict
    {**GOOD_RECORD, "transcript": "text"},                          # transcript must be dict/None
    {**GOOD_RECORD, "ocr": ["segments"]},                           # ocr must be dict/None
])
def test_valid_checkpoint_rejects_malformed_records(tmp_path, bad):
    (tmp_path / "record.json").write_text(json.dumps(bad))
    assert not worker.valid_checkpoint(tmp_path)


def test_valid_checkpoint_accepts_well_formed_record(tmp_path):
    (tmp_path / "record.json").write_text(json.dumps(GOOD_RECORD))
    assert worker.valid_checkpoint(tmp_path)


def test_malformed_checkpoint_media_survives_quarantine(monkeypatch):
    lease()
    job = spool_one()
    (job / "record.json").write_text(json.dumps({**GOOD_RECORD, "errors": "not-a-list"}))  # parses, but bad
    monkeypatch.setattr(jobs, "record_attempt", db_down)
    for _ in range(worker.MAX_PERSIST_FAILURES):
        worker.run_once(LOOKUP)
    dead = worker.dead_dir() / job.name
    assert (dead / "audio.webm").exists() and (dead / "frames").exists()
    assert (dead / "record.json.corrupt").exists() and not (dead / "record.json").exists()


def test_corrupt_or_backoff_failures_json_never_blocks_other_jobs(monkeypatch):
    monkeypatch.setattr(worker, "RETRY_BACKOFF_S", 3600)
    lease(3)
    a, b, c = spool_one("R0"), spool_one("R1"), spool_one("R2")
    (a / "failures.json").write_text(json.dumps({"count": 1, "last": time.time(), "errors": []}))  # backing off
    (b / "failures.json").write_text("garbage")                                                  # corrupt
    assert worker.drain(LOOKUP) == 2
    assert a.exists() and not b.exists() and not c.exists()
    assert n_attempts("R1") == 1 and n_attempts("R2") == 1


# ---------------------------------------------------------------- crash recovery

def test_crash_after_commit_recovery_is_idempotent(monkeypatch):
    lease()
    job = spool_one()
    monkeypatch.setattr(worker.shutil, "rmtree", lambda *a, **k: None)      # "crash" before deleting the job
    worker.process_job(job, LOOKUP)
    monkeypatch.setattr(worker.shutil, "rmtree", REAL_RMTREE)
    after_commit = state("R0")
    assert job.exists() and n_attempts("R0") == 1
    for _ in range(2):                                                       # recovery twice: idempotent
        info = worker.recover()
    assert not job.exists() and info["jobs_resumed"] == 0
    assert n_attempts("R0") == 1 and state("R0") == after_commit
    assert state("R0")["state"] != "queued" and state("R0")["lease_session"] is None


def test_rerunning_a_committed_job_restores_state_instead_of_stranding(monkeypatch):
    lease()
    job = spool_one()
    monkeypatch.setattr(worker.shutil, "rmtree", lambda *a, **k: None)
    worker.process_job(job, LOOKUP)
    committed = state("R0")
    jobs.mark_queued("R0", "ig-a")                                           # e.g. stale recovery mark
    monkeypatch.setattr(worker.shutil, "rmtree", REAL_RMTREE)
    out = worker.process_job(job, LOOKUP)
    assert out["duplicate"] and n_attempts("R0") == 1 and state("R0")["state"] == committed["state"]


def test_recovery_removes_partial_uploads_and_restores_orphans():
    lease(2)
    spool_one("R1")
    partial = worker.spool_dir() / ".incoming-abc"
    partial.mkdir()
    (partial / "audio.webm").write_bytes(b"x")
    jobs.mark_queued("R0", "ig-a")                       # crashed between mark and publication
    info = worker.recover()
    assert not partial.exists() and info["partial_uploads_removed"] == 1 and info["jobs_resumed"] == 1
    assert state("R0")["state"] != "queued" and state("R1")["state"] == "queued"


def test_failed_publication_reverts_queued_state_exactly(monkeypatch):
    lease()
    before = state("R0")
    real_rename = Path.rename

    def failing(self, target):
        if self.name.startswith(".incoming-"):
            raise OSError("disk full")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", failing)
    with pytest.raises(OSError):
        spool_one()
    monkeypatch.setattr(Path, "rename", real_rename)
    assert state("R0") == before                          # still in_progress, same lease
    assert not list(worker.spool_dir().glob(".incoming-*")) and worker.pending_jobs() == []


def test_reconcile_cannot_run_between_mark_and_publication(monkeypatch):
    lease()
    marked, go = threading.Event(), threading.Event()

    def slow_mark(rid, sid):
        snap = REAL_MARK_QUEUED(rid, sid)
        marked.set()
        go.wait(5)
        return snap

    monkeypatch.setattr(jobs, "mark_queued", slow_mark)
    t_spool = threading.Thread(target=spool_one)
    t_spool.start()
    assert marked.wait(5)
    result = {}
    t_rec = threading.Thread(target=lambda: result.update(n=worker.reconcile()))
    t_rec.start()
    time.sleep(0.3)
    assert t_rec.is_alive()                               # blocked: publication still in progress
    go.set()
    t_spool.join(5)
    t_rec.join(5)
    assert result["n"] == 0 and state("R0")["state"] == "queued" and len(worker.pending_jobs()) == 1


# ---------------------------------------------------------------- sessions + backpressure

def test_session_not_finalized_while_jobs_queued_and_never_abandoned_while_draining():
    lease(2)
    spool_one("R0")
    assert jobs.finish_session("ig-a") == 1               # R1 never uploaded → released
    p = jobs.session_progress("ig-a")
    assert p["capture_finished"] and not p["finalized"] and p["queued"] == 1
    assert jobs.heartbeat("ig-a")                          # extension keeps heart-beating while draining
    far_future = time.time() + 10 * jobs.STALE_SESSION_S
    other = jobs.create_session("ig-b", "ig", 10, now=far_future)   # triggers stale reclaim
    assert "R0" not in [i["reel_id"] for i in other]       # queued Reel never re-leased
    assert not jobs.session_progress("ig-a")["finalized"] and state("R0")["state"] == "queued"
    worker.drain(LOOKUP)
    assert jobs.session_progress("ig-a")["finalized"] and jobs.session_progress("ig-a")["queued"] == 0


def test_backpressure_refuses_upload_without_spooling(monkeypatch):
    monkeypatch.setattr(config, "ASYNC_PROCESSING", True)
    monkeypatch.setattr(worker, "MAX_BACKLOG", 1)
    monkeypatch.setattr(worker, "BACKPRESSURE_TIMEOUT_S", 0.2)
    lease(3)
    spool_one("R0"), spool_one("R1")                       # backlog 2 > max 1
    before = state("R2")
    r = TestClient(app).post("/api/process", data={"meta": json.dumps(cap("R2", playback="unavailable"))})
    assert r.status_code == 503 and "overloaded" in r.text
    assert len(worker.pending_jobs()) == 2 and state("R2") == before


def test_async_api_end_to_end_with_real_clip(monkeypatch):
    monkeypatch.setattr(config, "ASYNC_PROCESSING", True)
    client = TestClient(app)
    s = client.get("/api/session", params={"source": "test", "limit": 1}).json()
    audio, frames, walls = clip_upload("TESTCLIP01")
    meta = {"reel_id": "TESTCLIP01", "source": "test", "session_id": s["session_id"], "playback": "ok",
            "video_duration_s": 12.41, "expected_s": 12.41, "captured_s": 12.41, "play_wall": PLAY_WALL,
            "rec_start_wall": PLAY_WALL, "frame_walls": walls, "frame_interval_ms": 1000}
    files = [("audio", ("a.webm", audio, "audio/webm"))] + [("frames", (f"f{i}.jpg", f, "image/jpeg"))
                                                             for i, f in enumerate(frames)]
    r = client.post("/api/process", data={"meta": json.dumps(meta)}, files=files).json()
    assert r["status"] == "queued"
    dup = client.post("/api/process", data={"meta": json.dumps(meta)}, files=files).json()
    assert dup["duplicate"] and len(worker.pending_jobs()) == 1             # no second job
    client.post(f"/api/session/{s['session_id']}/finish")
    st = client.get(f"/api/session/{s['session_id']}/status").json()
    assert st["queued"] == 1 and not st["finalized"]
    assert worker.drain(_lookup) == 1
    st = client.get(f"/api/session/{s['session_id']}/status").json()
    assert st["queued"] == 0 and st["finalized"] and worker.pending_jobs() == []
    with jobs.connect() as con:
        row = con.execute("SELECT status FROM attempts WHERE reel_id='TESTCLIP01'").fetchone()
    assert row["status"] == "ok"


def test_retry_dead_requeues_quarantined_job(monkeypatch):
    lease()
    job = spool_one()
    monkeypatch.setattr(jobs, "record_attempt", db_down)
    for _ in range(worker.MAX_PERSIST_FAILURES):
        worker.run_once(LOOKUP)
    monkeypatch.setattr(jobs, "record_attempt", REAL_RECORD_ATTEMPT)
    assert worker.retry_dead() == 1 and state("R0")["state"] == "queued"
    assert worker.drain(LOOKUP) == 1 and n_attempts("R0") == 1                # persisted from record.json
