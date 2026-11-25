"""Job tracking: resumability, duplicates, retries, recovery, canonical records, migration."""
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import config, jobs
from backend.app import app


@pytest.fixture(autouse=True)
def _iso(isolated_output):
    jobs._export_state.update(pending=0, last=0.0)
    return isolated_output


def posts(n=5):
    return [{"reel_id": f"R{i}", "source_url": f"https://www.instagram.com/reel/R{i}/", "source_index": i,
             "media_type": "reel", "saved_at": None} for i in range(n)]


def rec(rid, sid, status="ok", errors=(), playback="ok", detail=None, text="some words"):
    return {"reel_id": rid, "session_id": sid, "processed_at": "2026-11-23T00:00:00+00:00",
            "status": status, "errors": list(errors), "source_url": f"u/{rid}",
            "capture": {"playback": playback, "playback_detail": detail},
            "transcript": {"text": text if status != "failed" else ""}, "ocr": {"segments": []}}


def state(rid):
    with jobs.connect() as con:
        return dict(con.execute("SELECT * FROM reels WHERE reel_id=?", (rid,)).fetchone())


def canonical_lines():
    return jobs.read_jsonl(config.OUTPUT_JSONL)


# ---------------------------------------------------------------- import / leasing / resume

def test_import_is_idempotent_and_never_resets_state():
    assert jobs.import_posts("ig", posts(3)) == 3
    jobs.create_session("ig-a", "ig", 1)
    jobs.record_attempt(rec("R0", "ig-a"))
    assert jobs.import_posts("ig", posts(3)) == 0           # duplicate import adds nothing
    assert state("R0")["state"] == "done" and state("R0")["attempts"] == 1


def test_sessions_do_not_share_leases_and_resume_skips_done():
    jobs.import_posts("ig", posts(5))
    a = jobs.create_session("ig-a", "ig", 2)
    b = jobs.create_session("ig-b", "ig", 2)
    assert [i["reel_id"] for i in a] == ["R0", "R1"] and [i["reel_id"] for i in b] == ["R2", "R3"]
    jobs.record_attempt(rec("R0", "ig-a"))
    jobs.record_attempt(rec("R1", "ig-a"))
    jobs.finish_session("ig-a")
    assert jobs.finish_session("ig-b") == 2                   # R2, R3 never reported → back to pending
    c = jobs.create_session("ig-c", "ig", 10)
    assert [i["reel_id"] for i in c] == ["R2", "R3", "R4"]   # done Reels are not re-queued


def test_interrupted_session_is_reclaimed_only_when_stale():
    jobs.import_posts("ig", posts(3))
    t0 = time.time() - 2 * jobs.STALE_SESSION_S
    jobs.create_session("ig-old", "ig", 2, now=t0)            # crashed: no heartbeat since t0
    assert [i["reel_id"] for i in jobs.create_session("ig-new", "ig", 10)] == ["R0", "R1", "R2"]


def test_heartbeat_keeps_long_running_session_alive():
    jobs.import_posts("ig", posts(3))
    t0 = time.time() - 2 * jobs.STALE_SESSION_S
    jobs.create_session("ig-long", "ig", 2, now=t0)          # started long ago, no Reel completed since…
    assert jobs.heartbeat("ig-long")                          # …but the extension is still heart-beating
    assert [i["reel_id"] for i in jobs.create_session("ig-other", "ig", 10)] == ["R2"]
    assert state("R0")["lease_session"] == "ig-long"


@pytest.mark.parametrize("mode", ["all", "eval"])
def test_unprocessed_reprocess_lease_restores_prior_state(mode):
    jobs.import_posts("ig", posts(3))
    jobs.create_session("ig-a", "ig", 3)
    jobs.record_attempt(rec("R0", "ig-a"))                                         # done
    jobs.record_attempt(rec("R1", "ig-a", "failed", ["stalled"], playback="stalled"))  # needs_retry
    jobs.finish_session("ig-a")                                                    # R2 → pending
    jobs.create_session("ig-b", "ig", 3, mode=mode, reel_ids=["R0", "R1", "R2"])
    assert jobs.finish_session("ig-b") == 3                                        # nothing reported
    assert [state(r)["state"] for r in ("R0", "R1", "R2")] == ["done", "needs_retry", "pending"]
    # stale-session reclaim restores prior states too
    jobs.create_session("ig-c", "ig", 3, mode="all", now=time.time() - 2 * jobs.STALE_SESSION_S)
    jobs.create_session("ig-d", "ig", 0)
    assert [state(r)["state"] for r in ("R0", "R1", "R2")] == ["done", "needs_retry", "pending"]


def test_heartbeat_after_finish_is_rejected():
    jobs.import_posts("ig", posts(1))
    jobs.create_session("ig-a", "ig", 1)
    jobs.finish_session("ig-a")
    assert jobs.heartbeat("ig-a") is False


# ---------------------------------------------------------------- duplicates & retries

def test_duplicate_upload_is_idempotent():
    jobs.import_posts("ig", posts(1))
    jobs.create_session("ig-a", "ig", 1)
    first = jobs.record_attempt(rec("R0", "ig-a", "failed", ["stalled"], playback="stalled"))
    again = jobs.record_attempt(rec("R0", "ig-a", "ok"))
    assert again["duplicate"] and again["attempt_id"] == first["attempt_id"]
    assert state("R0")["attempts"] == 1 and state("R0")["state"] == "needs_retry"
    assert jobs.status()["attempts"] == 1


def test_retries_then_exhausted_with_best_attempt_canonical():
    jobs.import_posts("ig", posts(1))
    outs = []
    for i in range(jobs.MAX_ATTEMPTS):
        jobs.create_session(f"ig-{i}", "ig", 1)
        status = "partial" if i == 1 else "failed"
        outs.append(jobs.record_attempt(rec("R0", f"ig-{i}", status, ["playback_stalled"], playback="stalled")))
    assert [o["state"] for o in outs] == ["needs_retry", "needs_retry", "exhausted"]
    assert jobs.create_session("ig-x", "ig", 10) == []                     # exhausted is not re-queued
    jobs.export_canonical()
    (line,) = canonical_lines()
    assert line["status"] == "partial" and line["processing"]["canonical_attempt_id"] == outs[1]["attempt_id"]
    assert len(line["processing"]["history"]) == 3


def test_free_retries_do_not_consume_attempts():
    jobs.import_posts("ig", posts(1))
    for i, pb in enumerate(["auth_required", "aborted", "capture_lost"]):
        jobs.create_session(f"ig-{i}", "ig", 1)
        out = jobs.record_attempt(rec("R0", f"ig-{i}", "failed", [pb], playback=pb))
        assert out["outcome"] == "retry_free" and out["state"] == "pending"
    assert state("R0")["attempts"] == 0


def test_unavailable_confirmed_twice_becomes_permanent_but_rate_limit_does_not():
    jobs.import_posts("ig", posts(2))
    for i in range(2):
        jobs.create_session(f"ig-{i}", "ig", 2)
        a = jobs.record_attempt(rec("R0", f"ig-{i}", "failed", ["unavailable"], playback="unavailable",
                                    detail="Page not found"))
        b = jobs.record_attempt(rec("R1", f"ig-{i}", "failed", ["unavailable"], playback="unavailable",
                                    detail="rate limited by site"))
    assert a["state"] == "failed_permanent" and b["state"] == "needs_retry"


def test_acceptable_partial_is_done_but_muted_partial_retries():
    jobs.import_posts("ig", posts(2))
    jobs.create_session("ig-a", "ig", 2)
    assert jobs.record_attempt(rec("R0", "ig-a", "partial", ["truncated_at_capture_limit: 180 of 240"]))["state"] == "done"
    assert jobs.record_attempt(rec("R1", "ig-a", "partial", ["audio_muted: x"]))["state"] == "needs_retry"


def test_reprocess_never_regresses_a_done_reel():
    jobs.import_posts("ig", posts(1))
    jobs.create_session("ig-a", "ig", 1)
    good = jobs.record_attempt(rec("R0", "ig-a", "ok", text="a long good transcript"))
    jobs.create_session("ig-b", "ig", 1, mode="all")
    bad = jobs.record_attempt(rec("R0", "ig-b", "failed", ["stalled"], playback="stalled"))
    assert bad["state"] == "done" and bad["canonical_attempt_id"] == good["attempt_id"]
    jobs.export_canonical()
    (line,) = canonical_lines()
    assert line["status"] == "ok" and [h["status"] for h in line["processing"]["history"]] == ["ok", "failed"]


def test_canonical_export_one_line_per_reel_and_history_complete():
    jobs.import_posts("ig", posts(3))
    for s in ("ig-a", "ig-b"):
        jobs.create_session(s, "ig", 3, mode="all")
        for r in ("R0", "R1", "R2"):
            jobs.record_attempt(rec(r, s))
    jobs.export_canonical()
    assert [l["reel_id"] for l in canonical_lines()] == ["R0", "R1", "R2"]
    assert len(jobs.read_jsonl(jobs.attempts_jsonl())) == 6


def test_jsonl_survives_unicode_line_separators():
    """Real case: an Instagram export caption (raw mojibake) contained U+0085, which str.splitlines() splits on."""
    nel, lsep, psep = chr(0x85), chr(0x2028), chr(0x2029)
    jobs.import_posts("ig", posts(1))
    jobs.create_session("ig-a", "ig", 1)
    tricky = f"mojibake{nel}text | sep{lsep}line | para{psep}end"
    r = rec("R0", "ig-a")
    r["metadata"] = {"raw": {"caption": tricky}}
    jobs.record_attempt(r)
    jobs.export_canonical()
    for path in (jobs.attempts_jsonl(), config.OUTPUT_JSONL):
        text = path.read_text(encoding="utf-8")
        assert not any(c in text for c in (nel, lsep, psep))                  # escaped on disk
        assert len(text.splitlines()) == len(jobs.read_jsonl(path)) == 1      # even naive readers agree
        assert jobs.read_jsonl(path)[0]["metadata"]["raw"]["caption"] == tricky  # value round-trips exactly


def test_export_is_debounced():
    jobs.import_posts("ig", posts(12))
    jobs.create_session("ig-a", "ig", 12)
    jobs.record_attempt(rec("R0", "ig-a"))                    # first export (time-based)
    for i in range(1, jobs.EXPORT_EVERY_N):
        jobs.record_attempt(rec(f"R{i}", "ig-a"))
    assert len(canonical_lines()) == 1                        # not rewritten on every attempt
    jobs.record_attempt(rec(f"R{jobs.EXPORT_EVERY_N}", "ig-a"))  # Nth pending attempt triggers a refresh
    assert len(canonical_lines()) == jobs.EXPORT_EVERY_N + 1
    jobs.finish_session("ig-a")                                # finishing always exports
    assert not config.OUTPUT_JSONL.with_suffix(".jsonl.tmp").exists()


# ---------------------------------------------------------------- migration

def write_phase1():
    lines = [rec("R0", "ig-p1a", "partial", ["audio_muted"]), rec("R0", "ig-p1b", "ok"), rec("R1", "ig-p1b", "ok")]
    config.OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_JSONL.write_text("".join(json.dumps(l) + "\n" for l in lines))
    return config.OUTPUT_JSONL.read_bytes()


def test_migration_backs_up_and_is_rerunnable():
    original = write_phase1()
    out = jobs.migrate_phase1(sources={"ig": posts(2)})
    assert out["imported"] == 3
    assert Path(out["backup"]).read_bytes() == original          # untouched copy of Phase-1 data
    assert [l["reel_id"] for l in canonical_lines()] == ["R0", "R1"]
    assert jobs.get_meta("phase1_migration_in_progress") is None
    again = jobs.migrate_phase1(sources={"ig": posts(2)})       # re-run: reads the backup, imports nothing
    assert again["imported"] == 0 and again["skipped"] == 3 and jobs.status()["attempts"] == 3


@pytest.mark.parametrize("resume_explicitly", [False, True])
def test_interrupted_migration_resumes_without_loss_or_duplicates(monkeypatch, resume_explicitly):
    original = write_phase1()
    real, calls = jobs.record_attempt, {"n": 0}

    def flaky(record, export=True):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated crash mid-migration")
        return real(record, export)

    monkeypatch.setattr(jobs, "record_attempt", flaky)
    with pytest.raises(KeyboardInterrupt):
        jobs.migrate_phase1(sources={"ig": posts(2)})
    monkeypatch.setattr(jobs, "record_attempt", real)
    assert jobs.status()["attempts"] == 1
    assert jobs.get_meta("phase1_migration_in_progress") is not None
    # Simulate the worst case: an export already replaced enriched_reels.jsonl after the crash.
    config.OUTPUT_JSONL.write_text(json.dumps({"reel_id": "R0", "processing": {}}) + "\n")
    (backup,) = (config.OUTPUT_DIR / "backup").glob("*.jsonl")
    assert backup.read_bytes() == original

    # Re-running the normal command (no path) must find the backup via the in-progress marker.
    out = jobs.migrate_phase1(backup if resume_explicitly else None, sources={"ig": posts(2)})
    assert out["backup"] == str(backup)
    assert out["imported"] == 2 and out["skipped"] == 1 and jobs.status()["attempts"] == 3
    assert [l["reel_id"] for l in canonical_lines()] == ["R0", "R1"]
    assert jobs.get_meta("phase1_migration_in_progress") is None
    assert len(list((config.OUTPUT_DIR / "backup").glob("*.jsonl"))) == 1   # no second backup of the garbage


# ---------------------------------------------------------------- through the API

def test_api_duplicate_post_is_not_reprocessed(monkeypatch):
    client = TestClient(app)
    s = client.get("/api/session", params={"source": "test", "limit": 1}).json()
    meta = {"reel_id": "TESTMISSING", "source": "test", "session_id": s["session_id"], "playback": "unavailable",
            "playback_detail": "Page not found"}
    r1 = client.post("/api/process", data={"meta": json.dumps(meta)}).json()
    called = []
    monkeypatch.setattr("backend.record.build_record", lambda *a, **k: called.append(1))
    r2 = client.post("/api/process", data={"meta": json.dumps(meta)}).json()
    assert r2["duplicate"] and r2["attempt_id"] == r1["attempt_id"] and not called


def test_api_session_heartbeat_finish_cycle():
    client = TestClient(app)
    s = client.get("/api/session", params={"source": "test", "limit": 2}).json()
    sid = s["session_id"]
    assert client.post(f"/api/session/{sid}/heartbeat").json()["ok"]
    assert client.post(f"/api/session/{sid}/finish").json()["released"] == 2
    again = client.get("/api/session", params={"source": "test", "limit": 2}).json()
    assert [i["reel_id"] for i in again["items"]] == [i["reel_id"] for i in s["items"]]
    assert client.get("/api/status").json()["reels"]["test"]["in_progress"] == 2
