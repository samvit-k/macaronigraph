"""Phase 3 incremental indexing: new / unchanged / improved / failed Reels, crash safety, orphans, error
preservation, idle gating, interruption, verified rebuilds, retirement of replaced collections, and
writer-lock shutdown safety. Everything runs in a temp OUTPUT_DIR with a deterministic hashing embedder."""
import fcntl
import io
import threading
import time

import pytest

from backend import config, embedder, indexer, jobs, worker
from helpers import canonical_digest
from test_chunking import make_record, oseg, seg

@pytest.fixture(autouse=True)
def idx(isolated_output, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    monkeypatch.setattr(indexer, "_requests", [])
    jobs._export_state.update(pending=0, last=0.0)
    yield isolated_output
    indexer.release_writer()
    assert isolated_output in config.chroma_dir().parents

_n = {"i": 0}

def add(rid, words="sourdough bread baking", status="ok", ocr=None, extra_transcript=""):
    """Persist an attempt exactly as the worker does (jobs.record_attempt); a new session per attempt."""
    _n["i"] += 1
    tr = [seg(0, 4, f"{words}. {extra_transcript}".strip())] if status != "failed" else []
    rec = make_record(tr, ocr or [], rid=rid, status=status, caption=f"About {words} #tips")
    rec.update(session_id=f"instagram-s{_n['i']}", processed_at=f"2026-11-23T00:00:{_n['i'] % 60:02d}+00:00",
               errors=[] if status != "failed" else ["capture_error: boom"], warnings=[])
    if status == "failed":
        rec["capture"]["playback"] = "error"
    return jobs.record_attempt(rec, export=False)

def emb():
    return embedder.for_spec(embedder.current_spec())

def active_col():
    return indexer.get_collection(indexer.active_collection())

def ids_for(col, rid):
    return sorted(col.get(where={"reel_id": rid}, include=[])["ids"])

# ---------------------------------------------------------------- basic incremental behaviour

def test_new_reels_are_indexed_and_searchable():
    add("A", "sourdough bread baking")
    add("B", "python asyncio event loop")
    out = indexer.sync_all()
    assert out["build"]["state"] == "activated"
    st = indexer.status()
    assert (st["eligible"], st["indexed"], st["pending"], st["errors"]) == (2, 2, 0, 0)
    res = indexer.search("asyncio event loop", k=2)
    assert res["results"][0]["reel_id"] == "B" and res["results"][0]["passages"]
    assert indexer.verify(indexer.active_collection())["ok"]

def test_unchanged_reels_are_skipped_without_embedding():
    add("A")
    add("B", "python asyncio")
    indexer.sync_all()
    calls = emb().calls
    out = indexer.sync_all()
    assert out["active"]["skipped"] == 2 and out["active"]["embedded"] == 0
    assert emb().calls == calls and "build" not in out

def test_new_reel_after_initial_index_is_added_incrementally():
    add("A")
    indexer.sync_all()
    add("C", "robot arm inverse kinematics")
    out = indexer.sync_all()
    assert out["active"]["embedded"] == 1 and out["active"]["skipped"] == 1
    assert indexer.search("inverse kinematics", k=1)["results"][0]["reel_id"] == "C"

def test_improved_canonical_record_replaces_chunks_and_removes_obsolete_ids():
    add("A", "sourdough", ocr=[oseg(1, 2, "Old screen one"), oseg(10, 11, "Old screen two")])
    indexer.sync_all()
    col = active_col()
    assert "A:ocr:001" in ids_for(col, "A")
    r = add("A", "sourdough", ocr=[oseg(1, 2, "New single screen with a much longer description")])
    assert r["canonical_attempt_id"] == r["attempt_id"]          # the better attempt became canonical
    out = indexer.sync_all()
    assert out["active"]["embedded"] == 1
    after = ids_for(col, "A")
    assert "A:ocr:001" not in after and "A:ocr:000" in after
    assert col.get(ids=["A:ocr:000"])["documents"] == ["New single screen with a much longer description"]
    assert indexer.verify(indexer.active_collection())["ok"]

def test_failed_reel_is_never_indexed():
    add("A")
    add("F", "should not appear", status="failed")
    indexer.sync_all()
    assert ids_for(active_col(), "F") == [] and indexer.status()["eligible"] == 1

def test_sync_never_writes_the_job_tables():
    add("A")
    add("B", "python")
    before = canonical_digest()
    indexer.sync_all()
    indexer.sync_all(rebuild=True)
    assert canonical_digest() == before

def test_reel_processed_by_the_worker_is_picked_up_by_the_next_sync(monkeypatch):
    add("A")
    indexer.sync_all()
    jobs.import_posts("instagram", [{"reel_id": "W1", "source_url": "u/W1", "source_index": 9, "media_type": "reel",
                                     "saved_at": None}])
    jobs.create_session("instagram-w", "instagram", 5)
    rec = make_record([seg(0, 3, "lidar point cloud segmentation")], [], rid="W1")
    rec.update(session_id="instagram-w", processed_at="2026-11-23T01:00:00+00:00", errors=[], warnings=[])
    monkeypatch.setattr(worker, "build", lambda job, lookup: rec)
    job = worker.spool({"reel_id": "W1", "session_id": "instagram-w", "playback": "ok"}, io.BytesIO(b"a"),
                       [io.BytesIO(b"\xff\xd8x")])
    assert worker.process_job(job, lambda s, r: None)["status"] == "ok"
    out = indexer.sync_all()
    assert out["active"]["embedded"] == 1
    assert indexer.search("lidar point cloud", k=1)["results"][0]["reel_id"] == "W1"

# ---------------------------------------------------------------- crash safety / orphans / errors

def test_crash_between_upsert_and_state_commit_leaves_no_duplicates(monkeypatch):
    add("A", ocr=[oseg(1, 2, "Screen")])
    add("B", "python asyncio")
    real = indexer._write_state
    boom = {"armed": True}

    def flaky(collection, rid, *a, **k):
        if boom["armed"] and a[3] == "indexed":
            boom["armed"] = False
            raise RuntimeError("crash after upsert, before bookkeeping")
        return real(collection, rid, *a, **k)

    monkeypatch.setattr(indexer, "_write_state", flaky)
    with pytest.raises(RuntimeError):
        indexer.sync_all()
    assert indexer.active_collection() is None                    # nothing half-built is ever active
    monkeypatch.setattr(indexer, "_write_state", real)
    out = indexer.sync_all()                                      # resumes the same build
    assert out["build"]["state"] == "activated"
    v = indexer.verify(indexer.active_collection())
    assert v["ok"] and v["counts"]["chunks"] == v["counts"]["expected_chunks"] == active_col().count()

def test_orphan_reel_chunks_are_swept_even_when_nothing_needs_reembedding():
    add("A")
    indexer.sync_all()
    col = active_col()
    col.upsert(ids=["GHOST:ocr:000"], embeddings=[emb().encode_query("ghost").tolist()], documents=["ghost"],
               metadatas=[{"reel_id": "GHOST", "content_type": "ocr", "source": "instagram"}])
    assert not indexer.verify(indexer.active_collection())["ok"]
    calls = emb().calls
    out = indexer.sync_all()
    assert out["active"]["embedded"] == 0 and emb().calls == calls        # no re-embedding needed
    assert out["active"]["orphans_deleted"] == 1 and ids_for(col, "GHOST") == []
    assert indexer.verify(indexer.active_collection())["ok"]

def test_stray_chunk_of_an_indexed_reel_is_swept():
    add("A")
    indexer.sync_all()
    col = active_col()
    expected = ids_for(col, "A")
    col.upsert(ids=["A:ocr:099"], embeddings=[emb().encode_query("x").tolist()], documents=["stale screen"],
               metadatas=[{"reel_id": "A", "content_type": "ocr", "source": "instagram"}])
    out = indexer.sync_all()
    assert out["active"]["orphans_deleted"] == 1 and ids_for(col, "A") == expected

def test_failed_reindex_preserves_previous_embeddings(monkeypatch):
    add("A", "sourdough starter feeding")
    add("B", "python asyncio")
    indexer.sync_all()
    col = active_col()
    before = {i: d for i, d in zip(*[col.get(where={"reel_id": "A"})[k] for k in ("ids", "documents")])}
    add("A", "sourdough starter feeding", extra_transcript="an improved longer transcript " * 4)
    e = emb()
    real = e.encode_docs
    monkeypatch.setattr(e, "encode_docs", lambda texts: (_ for _ in ()).throw(RuntimeError("MPS out of memory")))
    out = indexer.sync_all()
    assert out["active"]["errors"] == 1 and not out["active"]["stopped"]  # completed sync → sweep ran
    after = {i: d for i, d in zip(*[col.get(where={"reel_id": "A"})[k] for k in ("ids", "documents")])}
    assert after == before                                        # last good chunks kept, untouched
    assert indexer.search("sourdough starter", k=1)["results"][0]["reel_id"] == "A"
    row = indexer.state_rows(indexer.active_collection())["A"]
    assert row["status"] == "error" and "MPS out of memory" in row["error"]
    assert indexer.status()["errors"] == 1
    assert not indexer.verify(indexer.active_collection())["ok"]  # strict: the error is visible
    monkeypatch.setattr(e, "encode_docs", real)
    out = indexer.sync_all()                                      # retried and repaired
    assert out["active"]["embedded"] == 1 and indexer.status()["errors"] == 0
    assert indexer.verify(indexer.active_collection())["ok"]

def test_indexing_error_blocks_activation_of_a_rebuild(monkeypatch):
    add("A")
    add("B", "python")
    indexer.sync_all()
    old = indexer.active_collection()
    real = indexer.chunking.build_chunks

    def broken(record, source):
        if record["reel_id"] == "B":
            raise ValueError("corrupt record")
        return real(record, source)

    monkeypatch.setattr(indexer.chunking, "build_chunks", broken)
    monkeypatch.setattr(embedder, "current_spec", lambda model_id=None: embedder.HashEmbedder(dim=32).spec)
    out = indexer.sync_all()
    assert out["build"]["state"] == "failed_verification"
    assert indexer.active_collection() == old
    assert any("indexing errors" in p for p in out["build"]["verification"]["problems"])

# ---------------------------------------------------------------- interruption

def test_stopping_before_a_batch_embeds_nothing_and_commits_nothing(monkeypatch):
    for r in "ABC":
        add(r, f"topic {r}")
    monkeypatch.setattr(indexer, "EMBED_BATCH_CHUNKS", 10_000)    # everything would go in one final batch
    checks = iter([True, True, True])                             # idle for the 3 Reel checks, then busy
    out = indexer.sync_all(should_continue=lambda: next(checks, False))
    assert out["build"]["state"] == "paused"
    assert emb().calls == 0                                       # no embedding batch after the failed check
    assert indexer.state_rows(out["build"]["collection"]) == {}
    out = indexer.sync_all()                                      # resumes on the next sync
    assert out["build"]["state"] == "activated" and indexer.status()["indexed"] == 3

def test_stopping_mid_sync_keeps_committed_reels_and_resumes_the_rest(monkeypatch):
    for r in "ABC":
        add(r, f"topic {r}")
    monkeypatch.setattr(indexer, "EMBED_BATCH_CHUNKS", 1)        # flush after every Reel
    checks = iter([True, True])                                  # reel A: loop check + flush check
    out = indexer.sync_all(should_continue=lambda: next(checks, False))
    coll = out["build"]["collection"]
    assert out["build"]["state"] == "paused" and emb().calls == 1
    assert sorted(indexer.state_rows(coll)) == ["A"]
    calls = emb().calls
    out = indexer.sync_all()
    assert out["build"]["state"] == "activated" and out["build"]["collection"] == coll
    assert emb().calls == calls + 2                               # only B and C embedded on resume

def test_capture_starting_mid_sync_stops_before_the_next_embedding(monkeypatch):
    for r in "ABC":
        add(r, f"topic {r}")
    indexer.sync_all()                                            # baseline index
    for r in "DEF":
        add(r, f"new topic {r}")
    monkeypatch.setattr(indexer, "EMBED_BATCH_CHUNKS", 1)
    real_load = indexer.load_record
    seen = []

    def load(aid):
        seen.append(aid)
        if len(seen) == 2:                                        # the user presses Start during the sync
            jobs.create_session("instagram-live", "instagram", 0)
        return real_load(aid)

    monkeypatch.setattr(indexer, "load_record", load)
    calls = emb().calls
    out = indexer.sync_all(should_continue=lambda: indexer.busy_reason() is None)
    assert out["active"]["stopped"] and emb().calls == calls + 1  # D committed; nothing embedded after Start
    assert indexer.status()["indexed"] == 4

# ---------------------------------------------------------------- idle gate

def test_busy_reason_covers_sessions_spool_and_quiet_period():
    add("A")
    now = time.time()
    jobs.create_session("instagram-x", "instagram", 0, now=now)
    assert "capture session" in indexer.busy_reason(now=now)
    jobs.finish_session("instagram-x")
    assert indexer.busy_reason() is None                          # no quiet period requested (between batches)
    assert "quiet period" in indexer.busy_reason(quiet_s=120)
    assert indexer.busy_reason(quiet_s=120, now=time.time() + 121) is None
    (worker.spool_dir() / ".incoming-x").mkdir(parents=True)     # an upload in progress
    assert "processing" in indexer.busy_reason(now=time.time() + 999)

def test_auto_loop_waits_for_idle_but_manual_sync_always_runs(monkeypatch):
    add("A")
    monkeypatch.setattr(config, "INDEX_IDLE_S", 0)
    monkeypatch.setattr(indexer, "POLL_S", 0.05)
    jobs.create_session("instagram-live", "instagram", 0)
    indexer.start()
    try:
        time.sleep(0.4)
        assert indexer.active_collection() is None and "capture session" in (indexer._state["busy"] or "")
        assert indexer.request_sync()["accepted"]                 # manual sync is always allowed
        for _ in range(100):
            if indexer.active_collection():
                break
            time.sleep(0.05)
        assert indexer.active_collection() is not None
    finally:
        assert indexer.stop()

# ---------------------------------------------------------------- verified rebuilds

def test_rebuild_catches_reels_processed_during_the_rebuild(monkeypatch):
    add("A")
    add("B", "python")
    real = indexer.sync_collection
    once = {"done": False}

    def sync_and_add(coll, should_continue=None):
        st = real(coll, should_continue)
        if not once["done"]:                                      # the worker stores a new Reel meanwhile
            once["done"] = True
            add("C", "robot gripper")
        return st

    monkeypatch.setattr(indexer, "sync_collection", sync_and_add)
    out = indexer.sync_all()
    assert out["build"]["state"] == "activated" and len(out["build"]["passes"]) >= 2
    assert out["build"]["verification"]["counts"]["eligible"] == 3
    assert ids_for(active_col(), "C")

def test_failed_verification_keeps_previous_index_active(monkeypatch):
    add("A")
    indexer.sync_all()
    old = indexer.active_collection()
    real = indexer.sync_collection

    def sync_and_corrupt(coll, should_continue=None):
        st = real(coll, should_continue)
        if coll != old:                                           # an orphan appears in the new collection
            indexer.get_collection(coll).upsert(ids=["GHOST:ocr:000"], embeddings=[[0.1] * 64], documents=["x"],
                                                metadatas=[{"reel_id": "GHOST"}])
        return st

    monkeypatch.setattr(indexer, "sync_collection", sync_and_corrupt)
    out = indexer.sync_all(rebuild=True)
    assert out["build"]["state"] == "failed_verification"
    assert indexer.active_collection() == old                    # still serving
    assert indexer.search("sourdough", k=1)["results"][0]["reel_id"] == "A"
    assert indexer.status()["build"]["state"] == "failed_verification"
    monkeypatch.setattr(indexer, "sync_collection", real)
    out = indexer.sync_all()                                      # the pending build resumes, repairs, passes
    assert out["build"]["state"] == "activated" and indexer.active_collection() != old

def test_signature_change_builds_new_collection_while_old_stays_active(monkeypatch):
    add("A")
    add("B", "python")
    indexer.sync_all()
    old = indexer.active_collection()
    monkeypatch.setattr(embedder, "current_spec", lambda model_id=None: embedder.HashEmbedder(dim=32).spec)
    monkeypatch.setattr(indexer, "EMBED_BATCH_CHUNKS", 1)
    checks = iter([True, True, True])                             # active: A, B checks; build: first Reel only
    out = indexer.sync_all(should_continue=lambda: next(checks, False))
    assert out["build"]["state"] == "paused"
    assert indexer.active_collection() == old                    # search unaffected during the rebuild
    assert indexer.search("python", k=1)["results"][0]["reel_id"] == "B"
    out = indexer.sync_all()
    new = indexer.active_collection()
    assert out["build"]["state"] == "activated" and new != old
    assert indexer.spec_of(indexer.get_collection(new))["dim"] == 32
    assert old in indexer.collections()                          # retired, not dropped yet (grace period)

# ---------------------------------------------------------------- retirement of replaced collections

def test_replaced_collection_is_not_dropped_while_a_search_uses_it():
    add("A")
    indexer.sync_all()
    old = indexer.active_collection()
    with indexer.reading() as pinned:                             # a search is in progress on the old index
        assert pinned == old
        out = indexer.sync_all(rebuild=True)                      # swap happens meanwhile
        assert out["build"]["state"] == "activated" and indexer.active_collection() != old
        assert indexer.cleanup_retired(now=time.time() + 10 * indexer.RETIRE_GRACE_S) == []
        assert indexer.get_collection(old).count() > 0            # still fully readable
    assert indexer.cleanup_retired(now=time.time()) == []         # grace period not over (other processes)
    assert indexer.cleanup_retired(now=time.time() + indexer.RETIRE_GRACE_S + 1) == [old]
    assert old not in indexer.collections() and indexer.state_rows(old) == {}

# ---------------------------------------------------------------- writer lock on shutdown

def _other_process_can_lock() -> bool:
    fd = open(config.chroma_dir() / ".writer.lock", "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except OSError:
        return False
    finally:
        fd.close()

def test_stop_keeps_writer_lock_while_indexer_thread_is_still_writing(monkeypatch):
    assert indexer.acquire_writer()
    release = threading.Event()
    inside = threading.Event()

    def long_write():
        with indexer.writing():
            inside.set()
            release.wait(5)

    t = threading.Thread(target=long_write)
    t.start()
    assert inside.wait(2)
    monkeypatch.setattr(indexer, "_thread", t)
    monkeypatch.setattr(indexer, "STOP_JOIN_S", 0.1)
    assert indexer.stop() is False                                # join timed out
    assert indexer.holds_writer() and not _other_process_can_lock()
    assert indexer.release_writer() is False                      # cannot be released under a running write
    release.set()
    t.join()
    assert indexer.stop() is True
    assert not indexer.holds_writer() and _other_process_can_lock()

def test_second_writer_is_refused_while_lock_is_held():
    assert indexer.acquire_writer()
    assert not _other_process_can_lock()
    assert indexer.release_writer()
    assert _other_process_can_lock()

# ---------------------------------------------------------------- search behaviour

def test_fragments_are_findable_but_down_weighted():
    add("A", "cooking", ocr=[oseg(1, 2, "zebra", quality="fragment")])
    add("B", "zebra stripes documentary")
    indexer.sync_all()
    res = indexer.search("zebra", k=10, group="none")
    frag = next(h for h in res["results"] if h["content_type"] == "ocr_fragment")
    assert frag["reel_id"] == "A" and frag["ocr_quality"] == "fragment"
    assert frag["score"] == pytest.approx(frag["similarity"] * config.FRAGMENT_WEIGHT, abs=1e-3)

def test_type_and_source_filters():
    add("A", "sourdough", ocr=[oseg(1, 2, "Flour water salt")])
    indexer.sync_all()
    res = indexer.search("flour", k=5, types=["transcript"], group="none")
    assert res["results"] and {h["content_type"] for h in res["results"]} == {"transcript"}
    assert indexer.search("flour", source="test")["results"] == []

def test_search_before_any_index_is_not_ready():
    with pytest.raises(indexer.IndexNotReady):
        indexer.search("anything")