"""Graph sync (mocked: in-memory store with the Neo4jStore interface + deterministic fake LLM; real SQLite and a
real temp Chroma index with the hashing embedder). Live Neo4j behaviour is in test_graph_neo4j_live.py."""
import hashlib
import json
import re
import threading
import time

import pytest

from backend import config, embedder, indexer, jobs
from backend.graph import extract as ex
from backend.graph import sync as gs
from backend.graph.store import GraphOffline
from test_chunking import make_record, seg
from test_indexer import add


class FakeStore:
    """Same interface as Neo4jStore; keeps whole Reel documents (one 'transaction' = one dict assignment)."""

    def __init__(self):
        self.docs, self.ready, self.offline, self.fail_write, self.fail_ready = {}, {}, False, None, False
        self.writes = 0

    def _check(self):
        if self.offline:
            raise GraphOffline("fake offline")

    def ensure_schema(self):
        self._check()

    def reel_hashes(self):
        self._check()
        return {r: d["graph_hash"] for r, d in self.docs.items()}

    def write_reel(self, doc):
        self._check()
        if self.fail_write:
            raise self.fail_write
        self.docs[doc["reel"]["reel_id"]] = json.loads(json.dumps(doc))
        self.ready[doc["reel"]["reel_id"]] = doc["retrieval_ready"]
        self.writes += 1

    def remove_reel(self, rid):
        self._check()
        self.docs.pop(rid, None)
        self.ready.pop(rid, None)

    def set_ready(self, rid, ready):
        self._check()
        if self.fail_ready:
            raise GraphOffline("dropped during readiness update")
        self.ready[rid] = ready

    def concepts(self):
        return {c["key"] for d in self.docs.values() for c in d["concepts"]}


TERMS = ["Triton", "CUDA", "sourdough", "asyncio", "Python", "LiDAR", "robot"]


class FakeLLM:
    """Proposes every known term that occurs in a passage; one relation when a passage says 'built on top of'."""
    version = "x-fake-1"

    def __init__(self):
        self.calls, self.fail = 0, None

    def generate(self, system, user, max_tokens=0):
        self.calls += 1
        if self.fail:
            raise self.fail
        lines = re.findall(r"^(\d+)\. \([^)]*\) (.*)$", user.split("PASSAGES:")[1], flags=re.M)
        concepts, rels = [], []
        for n, text in lines:
            for t in TERMS:
                if re.search(rf"\b{re.escape(t)}\b", text, flags=re.I):
                    concepts.append({"n": t, "t": "technology", "s": t, "p": [int(n)]})
            m = re.search(r"(Triton is built on top of CUDA)", text)
            if m:
                rels.append({"s": "Triton", "p": "DEPENDS_ON", "o": "CUDA", "i": int(n), "q": m.group(1)})
        return json.dumps({"c": concepts, "r": rels}), {"finish_reason": "stop"}

    def unload(self):
        pass


@pytest.fixture(autouse=True)
def env(isolated_output, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    monkeypatch.setattr(gs, "_requests", [])
    monkeypatch.setattr(gs, "_request_log", {})
    jobs._export_state.update(pending=0, last=0.0)
    # Hard guard installed outside monkeypatch (a monkeypatch rollback cannot remove it): any database or
    # Chroma access outside the temp OUTPUT_DIR fails the test instead of touching production data.
    real_db, real_chroma = jobs.db_path, config.chroma_dir

    def guarded(fn):
        def inner():
            p = fn()
            assert isolated_output in p.parents, f"test reached production path {p}"
            return p
        return inner

    jobs.db_path, config.chroma_dir = guarded(real_db), guarded(real_chroma)
    try:
        yield
    finally:
        jobs.db_path, config.chroma_dir = real_db, real_chroma
        indexer.release_writer()


def triton_reel(rid="A", extra=""):
    return add(rid, f"Triton is built on top of CUDA {extra}".strip())


def run(store, llm, **kw):
    return gs.sync(store=store, llm=llm, **kw)


def digest_canonical():
    with jobs.connect() as con:
        h = hashlib.sha256()
        for t in ("reels", "attempts", "sessions"):
            for row in con.execute(f"SELECT * FROM {t} ORDER BY rowid"):
                h.update(repr(tuple(row)).encode())
        return h.hexdigest()


# ---------------------------------------------------------------- ingest / skip / update

def test_initial_ingest_links_passages_to_chroma_chunks_with_provenance():
    triton_reel("A")
    add("B", "asyncio in Python")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    st = run(store, llm)
    assert (st["written"], st["ready"], st["extracted"]) == (2, 2, 2)
    doc = store.docs["A"]
    col = indexer.get_collection(indexer.active_collection())
    assert {p["chunk_id"] for p in doc["passages"]} == set(col.get(where={"reel_id": "A"}, include=[])["ids"])
    # the claim is stated in the caption and in speech: one assertion per evidence passage
    assert sorted(x["chunk_id"].split(":")[1] for x in doc["assertions"]) == ["caption", "transcript"]
    a = _the_assertion(store)
    assert (a["subject"], a["predicate"], a["object"], a["status"]) == ("triton", "DEPENDS_ON", "cuda", "proposed")
    assert a["provenance"] == "quote_verified" and a["chunk_id"] in {p["chunk_id"] for p in doc["passages"]}
    assert all(m["chunk_id"] in {p["chunk_id"] for p in doc["passages"]} for m in doc["mentions"])
    assert gs.states()["A"]["status"] == "ready"


def test_unchanged_reels_are_skipped_without_llm_or_rewrite():
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    run(store, llm)
    calls, writes = llm.calls, store.writes
    st = run(store, llm)
    assert st["skipped"] == 1 and llm.calls == calls and store.writes == writes


def test_improved_canonical_record_is_reconciled():
    triton_reel("A")
    add("B", "sourdough")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    run(store, llm)
    add("B", "sourdough and asyncio with Python", extra_transcript="longer " * 10)
    indexer.sync_all()
    st = run(store, llm)
    assert (st["extracted"], st["written"], st["skipped"]) == (1, 1, 1)
    assert {"asyncio", "python"} <= {c["key"] for c in store.docs["B"]["concepts"]}


def test_repeated_sync_creates_no_duplicates():
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    for _ in range(3):
        run(store, llm)
    doc = store.docs["A"]
    assert len(doc["assertions"]) == len({a["id"] for a in doc["assertions"]}) == 2   # caption + speech
    assert len(doc["mentions"]) == len({(m["chunk_id"], m["key"]) for m in doc["mentions"]})


def test_test_clips_are_excluded_from_the_production_graph():
    triton_reel("A")
    rec = make_record([seg(0, 3, "Triton test clip")], [], rid="TEST01")
    rec.update(session_id="test-1", processed_at="2026-11-23T00:00:00+00:00", errors=[], warnings=[])
    jobs.record_attempt(rec, export=False)
    indexer.sync_all()
    store = FakeStore()
    run(store, FakeLLM())
    assert set(store.docs) == {"A"}


def test_sync_never_modifies_canonical_tables():
    triton_reel("A")
    indexer.sync_all()
    before = digest_canonical()
    run(FakeStore(), FakeLLM())
    assert digest_canonical() == before


# ---------------------------------------------------------------- failures and recovery

def test_failed_extraction_keeps_previous_graph_knowledge():
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    run(store, llm)
    old = json.dumps(store.docs["A"], sort_keys=True)
    old_state = gs.states()["A"]
    triton_reel("A", extra="and much more about GPUs " * 3)
    llm.fail = RuntimeError("model crashed")
    st = run(store, llm)
    assert st["extract_errors"] == 1
    assert json.dumps(store.docs["A"], sort_keys=True) == old                     # graph untouched
    s = gs.states()["A"]
    assert s["status"] == "error" and s["graph_hash"] == old_state["graph_hash"]
    assert s["canonical_attempt_id"] == old_state["canonical_attempt_id"]       # still describes the graph
    llm.fail = None
    indexer.sync_all()                                                           # Chroma catches up too
    st = run(store, llm)
    assert st["written"] == 1 and gs.states()["A"]["status"] == "ready"


def test_failed_graph_transaction_keeps_previous_graph_and_retries():
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    run(store, llm)
    old = json.dumps(store.docs["A"], sort_keys=True)
    triton_reel("A", extra="plus Python")
    indexer.sync_all()
    store.fail_write = RuntimeError("constraint violation")
    st = run(store, llm)
    assert st["write_errors"] == 1 and json.dumps(store.docs["A"], sort_keys=True) == old
    store.fail_write = None
    calls = llm.calls
    st = run(store, llm)
    assert st["written"] == 1 and llm.calls == calls                             # reused cached extraction


def test_crash_after_commit_before_state_is_recorded_without_rewrite(monkeypatch):
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    real = gs._commit_state_if_reviews_unchanged
    monkeypatch.setattr(gs, "_commit_state_if_reviews_unchanged",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        run(store, llm)
    assert "A" in store.docs and "A" not in gs.states()
    monkeypatch.setattr(gs, "_commit_state_if_reviews_unchanged", real)
    writes = store.writes
    st = run(store, llm)
    assert store.writes == writes and gs.states()["A"]["status"] == "ready" and st["written"] == 0


def test_interrupted_sync_resumes_at_reel_boundary():
    for r in "ABC":
        triton_reel(r)
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    checks = iter([True])
    st = run(store, llm, should_continue=lambda: next(checks, False))
    assert st["stopped"] and len(store.docs) == 1
    st = run(store, llm)
    assert len(store.docs) == 3 and st["skipped"] == 1 and st["written"] == 2


def test_neo4j_offline_still_fills_the_extraction_cache_then_catches_up():
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    store.offline = True
    st = run(store, llm)
    assert not st["online"] and st["extracted"] == 1 and store.docs == {}
    assert indexer.search("Triton", k=1)["results"][0]["reel_id"] == "A"      # semantic search unaffected
    store.offline = False
    calls = llm.calls
    st = run(store, llm)
    assert st["written"] == 1 and st["from_cache"] == 1 and llm.calls == calls


def test_ineligible_reel_is_removed(monkeypatch):
    triton_reel("A")
    add("B", "sourdough")
    indexer.sync_all()
    store = FakeStore()
    run(store, FakeLLM())
    real = gs.eligible
    monkeypatch.setattr(gs, "eligible", lambda: {k: v for k, v in real().items() if k != "B"})
    st = run(store, FakeLLM())
    assert st["removed"] == 1 and set(store.docs) == {"A"} and "B" not in gs.states()


# ---------------------------------------------------------------- Chroma readiness

def test_reel_awaits_the_index_then_is_promoted_without_llm():
    triton_reel("A")
    store, llm = FakeStore(), FakeLLM()
    st = run(store, llm)                                          # no Chroma index yet
    assert st["awaiting_index"] == 1 and store.ready["A"] is False
    indexer.sync_all()
    calls, writes = llm.calls, store.writes
    st = run(store, llm)
    assert st["promoted"] == 1 and store.ready["A"] is True and gs.states()["A"]["status"] == "ready"
    assert llm.calls == calls and store.writes == writes


def test_ready_reel_is_demoted_when_chroma_is_unavailable_and_promoted_on_recovery(monkeypatch):
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    run(store, llm)
    calls, writes = llm.calls, store.writes
    real_active, available = indexer.active_collection, {"on": False}
    monkeypatch.setattr(indexer, "active_collection", lambda: real_active() if available["on"] else None)
    st = run(store, llm)                                                         # index unavailable
    assert st["demoted"] == 1 and store.ready["A"] is False and gs.states()["A"]["status"] == "awaiting_index"
    available["on"] = True                                                       # index recovers
    st = run(store, llm)
    assert st["promoted"] == 1 and store.ready["A"] is True and gs.states()["A"]["status"] == "ready"
    assert llm.calls == calls and store.writes == writes                       # no LLM, no rewrite


def test_ready_reel_is_demoted_when_its_chunks_go_missing():
    triton_reel("A")
    indexer.sync_all()
    store = FakeStore()
    run(store, FakeLLM())
    col = indexer.get_collection(indexer.active_collection())
    col.delete(ids=col.get(where={"reel_id": "A"}, include=[])["ids"][:1])
    st = run(store, FakeLLM())
    assert st["demoted"] == 1 and gs.states()["A"]["status"] == "awaiting_index"


def test_neo4j_drop_during_readiness_update_does_not_abort_sync(monkeypatch):
    triton_reel("A")
    add("B", "sourdough")
    indexer.sync_all()
    store = FakeStore()
    run(store, FakeLLM())
    monkeypatch.setattr(indexer, "active_collection", lambda: None)
    store.fail_ready = True
    st = run(store, FakeLLM())
    assert not st["online"] and st["stopped"] is False and st["eligible"] == 2
    assert {s["status"] for s in gs.states().values()} == {"ready"}             # unchanged; retried next sync
    store.fail_ready = False
    st = run(store, FakeLLM())
    assert st["demoted"] == 2


# ---------------------------------------------------------------- idle gating

def test_manual_sync_during_capture_defers_extraction_unless_forced(monkeypatch):
    triton_reel("A")
    indexer.sync_all()
    jobs.create_session("instagram-live", "instagram", 0)
    store, llm = FakeStore(), FakeLLM()
    monkeypatch.setattr(gs, "Neo4jStore", lambda: store)
    monkeypatch.setattr(gs, "extractor_version", lambda model_id=None: FakeLLM.version)
    import backend.graph.llm as llm_mod
    monkeypatch.setattr(llm_mod, "MlxLLM", lambda model_id=None: llm)
    st = gs.request_sync(wait=True)
    assert st["deferred"] == 1 and llm.calls == 0                               # capture active: no LLM
    st = gs.request_sync(force=True, wait=True)
    assert st["extracted"] == 1 and llm.calls == 1 and "A" in store.docs


# ---------------------------------------------------------------- concurrency

def test_only_one_sync_at_a_time_and_manual_wait_reports_busy(monkeypatch):
    triton_reel("A")
    indexer.sync_all()
    hold, inside = threading.Event(), threading.Event()

    def holder():
        with gs.exclusive():
            inside.set()
            hold.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    assert inside.wait(2)
    with pytest.raises(gs.SyncBusy):
        run(FakeStore(), FakeLLM())
    monkeypatch.setattr(gs, "Neo4jStore", FakeStore)
    assert gs.request_sync(wait=True)["busy"] is True
    hold.set()
    t.join()
    assert run(FakeStore(), FakeLLM())["written"] == 1


def test_queued_request_meeting_a_running_sync_is_retried_not_dropped(monkeypatch):
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    monkeypatch.setattr(gs, "Neo4jStore", lambda: store)
    monkeypatch.setattr(gs, "extractor_version", lambda model_id=None: FakeLLM.version)
    import backend.graph.llm as llm_mod
    monkeypatch.setattr(llm_mod, "MlxLLM", lambda model_id=None: llm)
    monkeypatch.setattr(gs, "POLL_S", 30)                                       # no polling rescue
    monkeypatch.setattr(config, "GRAPH_AUTO", False)
    hold, inside = threading.Event(), threading.Event()

    def holder():
        with gs.exclusive():
            inside.set()
            hold.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    assert inside.wait(2)
    gs.start()
    try:
        r = gs.request_sync()
        rid = r["request_id"]
        for _ in range(50):
            if (gs.request_status(rid) or {}).get("reason"):
                break
            time.sleep(0.02)
        assert gs.request_status(rid)["status"] == "queued" and "already running" in gs.request_status(rid)["reason"]
        hold.set()                                                               # running sync finishes → wakes loop
        t.join()
        for _ in range(200):
            if gs.request_status(rid)["status"] in ("done", "failed"):
                break
            time.sleep(0.02)
        assert gs.request_status(rid)["status"] == "done" and "A" in store.docs
    finally:
        assert gs.stop()


# ---------------------------------------------------------------- reviews

def _the_assertion(store):
    """The assertion whose evidence is the spoken passage (the caption states the same claim separately)."""
    (a,) = [x for x in store.docs["A"]["assertions"] if ":transcript:" in x["chunk_id"]]
    return a


def test_review_survives_reel_refresh():
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    run(store, llm)
    a = _the_assertion(store)
    assert gs.put_review(a["subject"], a["predicate"], a["object"], a["chunk_id"], "accept", "checked") == "A"
    assert gs.states()["A"]["status"] == "stale"
    run(store, llm)
    assert _the_assertion(store)["status"] == "verified" and _the_assertion(store)["review_note"] == "checked"
    triton_reel("A", extra="for GPU kernels")                                   # improved record, same claim
    indexer.sync_all()
    run(store, llm)
    assert _the_assertion(store)["status"] == "verified"


def test_review_during_sync_is_never_marked_synchronized(monkeypatch):
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    run(store, llm)
    a = _the_assertion(store)
    triton_reel("A", extra="for GPU kernels")                                   # force a rewrite
    indexer.sync_all()
    real = gs.reviews_for

    def snapshot_then_review(ids):
        out = real(ids)                                                        # sync takes its snapshot…
        gs.put_review(a["subject"], a["predicate"], a["object"], a["chunk_id"], "reject", "wrong")  # …review lands
        return out

    monkeypatch.setattr(gs, "reviews_for", snapshot_then_review)
    st = run(store, llm)
    assert st["stale"] == 1 and st["ready"] == 0
    s = gs.states()["A"]
    assert s["status"] == "stale" and s["graph_hash"] is None                  # not marked synchronized
    assert _the_assertion(store)["status"] == "proposed"                       # graph has the pre-review doc
    monkeypatch.setattr(gs, "reviews_for", real)
    run(store, llm)
    assert _the_assertion(store)["status"] == "rejected" and gs.states()["A"]["status"] == "ready"


def test_review_changed_between_snapshot_and_commit_is_detected(monkeypatch):
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    run(store, llm)
    a = _the_assertion(store)
    gs.put_review(a["subject"], a["predicate"], a["object"], a["chunk_id"], "accept", None)
    real_write = store.write_reel

    def write_then_change(doc):
        real_write(doc)
        gs.put_review(a["subject"], a["predicate"], a["object"], a["chunk_id"], "reject", "changed my mind")

    store.write_reel = write_then_change
    run(store, llm)
    assert gs.states()["A"]["status"] == "stale"
    store.write_reel = real_write
    run(store, llm)
    assert _the_assertion(store)["status"] == "rejected"


def test_owner_of_chunk_ids():
    assert gs.owner_of("DtestReel02:transcript:003") == "DtestReel02"
    assert gs.owner_of("a-b_c:ocr_title:010") == "a-b_c"
    assert gs.owner_of("no-format") is None
    with pytest.raises(ValueError):
        gs.put_review("x", "USES", "y", "no-format", "accept", None)


# ---------------------------------------------------------------- extractor versions, recovery, no needless re-extraction

class NewLLM(FakeLLM):
    """A newer extractor whose only change is failure recovery: accepts FakeLLM's successful proposals."""
    version = "x-fake-2"
    compatible_versions = ["x-fake-1"]


class IncompatibleLLM(FakeLLM):
    version = "x-fake-3"
    compatible_versions = []


def _cache_rows():
    with jobs.connect() as con:
        return sorted((r["reel_id"], r["extractor_version"], r["status"])
                      for r in con.execute("SELECT * FROM graph_extraction"))


def test_compatible_proposals_are_reused_under_their_own_label_without_the_llm(monkeypatch):
    triton_reel("A")
    triton_reel("B", extra="for GPU kernels")
    indexer.sync_all()
    store = FakeStore()
    run(store, FakeLLM())
    new = NewLLM()
    st = run(store, new)
    assert (st["skipped"], st["extracted"], new.calls) == (2, 0, 0)               # nothing re-run, nothing rewritten
    monkeypatch.setattr(ex, "GROUNDING_RULES", "g-test")                           # grounding changes → re-ground only
    st = run(store, new)
    assert (st["from_cache"], st["extracted"], st["written"], new.calls) == (2, 0, 2, 0)
    for rid in ("A", "B"):                                                         # provenance is not relabelled
        doc = store.docs[rid]
        assert doc["reel"]["extractor_version"] == "x-fake-1"
        assert {m["extractor_version"] for m in doc["mentions"]} == {"x-fake-1"}
        assert {a["extractor_version"] for a in doc["assertions"]} == {"x-fake-1"}
        assert gs.states()[rid]["extractor_version"] == "x-fake-1"
    assert _cache_rows() == [("A", "x-fake-1", "ok"), ("B", "x-fake-1", "ok")]       # no copies under a new label
    other = IncompatibleLLM()
    st = run(store, other)
    assert st["extracted"] == 2 and other.calls == 2                               # incompatible → re-extracted
    assert store.docs["A"]["reel"]["extractor_version"] == "x-fake-3"


def test_only_failed_reels_are_re_extracted_and_a_second_sync_extracts_nothing():
    for rid in ("A", "B", "C"):
        triton_reel(rid)
    indexer.sync_all()
    store, old = FakeStore(), FakeLLM()
    real, calls = old.generate, {"n": 0}

    def counted(system, user, max_tokens=0):
        calls["n"] += 1
        if calls["n"] >= 3:                                                        # third Reel: truncated twice
            return '{"c":[', {"finish_reason": "length"}
        return real(system, user, max_tokens)
    old.generate = counted
    st = run(store, old)
    assert (st["extracted"], st["extract_errors"]) == (2, 1)
    failed = next(r for r, s in gs.states().items() if s["status"] == "error")
    new = NewLLM()
    st = run(store, new)
    assert (st["extracted"], st["from_cache"], st["skipped"], new.calls) == (1, 0, 2, 1)   # only the failed Reel
    assert gs.states()[failed]["extractor_version"] == "x-fake-2"
    st = run(store, new)
    assert (st["extracted"], st["skipped"], new.calls) == (0, 3, 1)                # steady state: no LLM at all


def test_deterministic_extraction_failure_is_not_retried_automatically():
    from backend.graph.extract import ExtractionError
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    llm.fail = ExtractionError("output truncated at 3000 tokens after retry and nothing complete to keep")
    assert run(store, llm)["extract_errors"] == 1 and llm.calls == 1
    llm.fail = None
    st = run(store, llm)
    assert (st["not_retried"], st["extracted"], llm.calls) == (1, 0, 1)            # same input → same output: skip
    assert "retry with force" in gs.states()["A"]["error"]
    st = run(store, llm, force=True)
    assert st["extracted"] == 1 and gs.states()["A"]["status"] == "ready"


def test_partial_extraction_is_flagged_on_the_reel_and_timing_is_reported():
    triton_reel("A")
    indexer.sync_all()
    store, llm = FakeStore(), FakeLLM()
    cut = '{"c":[{"n":"Triton","t":"technology","s":"Triton","p":[1]},{"n":"CU'
    llm.generate = lambda system, user, max_tokens=0: (cut, {"finish_reason": "length", "seconds": 2.0})
    st = run(store, llm)
    doc = store.docs["A"]
    assert doc["reel"]["extraction_partial"] is True and [c["key"] for c in doc["concepts"]] == ["triton"]
    assert set(st["timing"]) == {"prepare_s", "model_load_s", "llm_s", "ground_s", "chroma_s", "neo4j_s", "sqlite_s"}
    with jobs.connect() as con:
        stats = json.loads(con.execute("SELECT stats FROM graph_extraction").fetchone()[0])
    assert [a["outcome"] for a in stats["attempts"]] == ["truncated", "truncated"] and stats["seconds"] == 4.0
