"""Heavy-work lock and shared LLM (backend/models.py): thread and cross-process exclusion, timeouts,
re-entrancy, error recovery, idle unloading without races, shutdown, one copy, and integration with graph
extraction, index embedding and the background graph sync."""
import errno
import os
import subprocess
import sys
import threading
import time

import pytest

from backend import config, embedder, indexer, jobs, models
from backend.graph import sync as gs
from conftest import ROOT

@pytest.fixture(autouse=True)
def iso(isolated_output, monkeypatch):
    monkeypatch.setattr(models, "_shared", None)
    monkeypatch.setattr(models, "_llms", {})
    monkeypatch.setattr(models, "_last_use", 0.0)
    monkeypatch.setattr(models, "_closed", False)
    monkeypatch.setattr(models, "_timer", None)
    monkeypatch.setattr(models, "stats", {"created": 0, "unloads": 0, "evictions": 0})
    yield
    models.shutdown()

class FakeMlx:
    """Stands in for graph.llm.MlxLLM: counts weight loads; loading and generation can be slowed down."""
    instances = 0

    def __init__(self, model_id=None):
        FakeMlx.instances += 1
        self.model_id = model_id
        self._model, self.loads, self.delay, self.load_delay, self.load_seconds = None, 0, 0.0, 0.0, 0.0

    def generate(self, system, user, max_tokens=0):
        if self._model is None:
            time.sleep(self.load_delay)
            self._model, self.loads = object(), self.loads + 1
        time.sleep(self.delay)
        assert self._model is not None, "weights vanished during generation"
        return "ok", {"finish_reason": "stop"}

    def unload(self):
        self._model = None

@pytest.fixture
def fake_mlx(monkeypatch):
    from backend.graph import llm as graph_llm
    FakeMlx.instances = 0
    monkeypatch.setattr(graph_llm, "MlxLLM", FakeMlx)
    return FakeMlx

def _holder_process(seconds: float):
    """Another process holding the heavy lock on the same OUTPUT_DIR."""
    code = ("import sys, time; sys.path.insert(0, %r)\n"
            "from backend import config, models\n"
            "with models.heavy('cli graph sync'):\n"
            "    print('held', flush=True); time.sleep(%r)\n") % (str(ROOT), seconds)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "OUTPUT_DIR": str(config.OUTPUT_DIR)})
    assert p.stdout.readline().strip() == "held"
    return p

# ---------------------------------------------------------------- lock semantics

def test_reentrant_within_a_thread_and_reports_its_holder():
    with models.heavy("outer"):
        with models.heavy("inner"):
            assert models.holder()["purpose"] == "outer" and models.held_by_me()
        assert models.held_by_me()
    assert models.holder() is None and not models.held_by_me()

def test_threads_are_mutually_exclusive_with_timeout():
    entered, release = threading.Event(), threading.Event()

    def hold():
        with models.heavy("graph extraction"):
            entered.set()
            release.wait(5)
    t = threading.Thread(target=hold)
    t.start()
    entered.wait(5)
    t0 = time.monotonic()
    with pytest.raises(models.HeavyBusy, match="graph extraction"):
        with models.heavy("answer", timeout=0.3):
            pass
    assert 0.25 <= time.monotonic() - t0 < 2
    release.set()
    t.join(5)
    with models.heavy("answer", timeout=0.5):
        assert models.held_by_me()

def test_cross_process_mutual_exclusion_and_waiting():
    p = _holder_process(1.5)
    try:
        with pytest.raises(models.HeavyBusy, match="another process"):
            with models.heavy("answer", timeout=0.3):
                pass
        t0 = time.monotonic()
        with models.heavy("answer"):                           # blocking: waits for the other process
            waited = time.monotonic() - t0
        assert waited > 0.5
    finally:
        p.wait(10)

def test_exception_inside_releases_thread_and_process_locks():
    with pytest.raises(ValueError):
        with models.heavy("answer"):
            raise ValueError("boom")
    with models.heavy("again", timeout=0):                     # thread lock free
        pass
    p = _holder_process(0.1)                                    # file lock free for another process
    p.wait(10)
    assert p.returncode == 0

def test_unexpected_flock_error_propagates_without_retry_and_releases_the_thread_lock(monkeypatch):
    calls = []

    def bad_flock(fd, op):
        calls.append(op)
        raise OSError(errno.EIO, "I/O error")
    monkeypatch.setattr(models.fcntl, "flock", bad_flock)
    with pytest.raises(OSError, match="I/O error"):
        with models.heavy("answer", timeout=5):
            pass
    assert calls == [models.fcntl.LOCK_EX | models.fcntl.LOCK_NB]   # no retry, no unlock of an unheld lock
    assert models._tlock.acquire(timeout=0)
    models._tlock.release()

def test_failing_unlock_still_releases_the_thread_lock(monkeypatch):
    real = models.fcntl.flock

    def flaky(fd, op):
        if op == models.fcntl.LOCK_UN:
            raise OSError(errno.EIO, "unlock failed")
        return real(fd, op)
    monkeypatch.setattr(models.fcntl, "flock", flaky)
    with pytest.raises(OSError, match="unlock failed"):
        with models.heavy("answer"):
            pass
    assert models._tlock.acquire(timeout=0)                     # not leaked
    models._tlock.release()

# ---------------------------------------------------------------- shared LLM

def test_one_shared_instance_and_generation_only_inside_heavy(fake_mlx):
    a, b = models.shared_llm(), models.shared_llm()
    assert a is b and fake_mlx.instances == 1 and models.stats["created"] == 1
    with pytest.raises(RuntimeError, match="outside models.heavy"):
        a.generate("s", "u")
    with models.heavy("answer"):
        assert a.generate("s", "u")[0] == "ok"
    assert models.loaded()

def test_idle_unload_frees_weights_and_reload_uses_the_same_instance(fake_mlx):
    llm = models.shared_llm()
    with models.heavy("answer"):
        llm.generate("s", "u")
    assert not models.unload_if_idle(idle_s=60)                 # recently used: kept
    models._last_use = time.monotonic() - 120
    assert models.unload_if_idle(idle_s=60) and not models.loaded()
    with models.heavy("answer"):
        llm.generate("s", "u")                                  # reloads lazily, same object
    assert fake_mlx.instances == 1 and llm._inner.loads == 2 and models.stats["unloads"] == 1

def test_idle_unload_never_overlaps_a_running_generation(fake_mlx):
    llm = models.shared_llm()
    with models.heavy("warm"):
        llm.generate("s", "u")
    llm._inner.delay = 0.8
    started, result = threading.Event(), {}

    def gen():
        with models.heavy("answer"):
            started.set()
            result["out"] = llm.generate("s", "u")
    t = threading.Thread(target=gen)
    t.start()
    started.wait(5)
    models._last_use = time.monotonic() - 10_000               # looks idle, but a generation is running
    assert models.unload_if_idle(idle_s=1) is False
    t.join(5)
    assert result["out"][0] == "ok" and models.loaded()          # FakeMlx asserts weights never vanished

def test_unload_called_by_a_caller_is_deferred_not_immediate(fake_mlx):
    llm = models.shared_llm()
    with models.heavy("graph extraction"):
        llm.generate("s", "u")
    llm.unload()                                                # what graph sync calls at the end
    assert models.loaded()                                      # left to the idle timer

def test_shutdown_during_initial_model_loading_waits_then_frees(fake_mlx):
    llm = models.shared_llm()
    llm._inner.load_delay = 0.8                                 # first generate is still loading the weights
    started, result = threading.Event(), {}

    def gen():
        with models.heavy("answer"):
            started.set()
            result["out"] = llm.generate("s", "u")
    t = threading.Thread(target=gen)
    t.start()
    started.wait(5)
    t0 = time.monotonic()
    models.shutdown()                                           # must wait for the load + generation
    assert time.monotonic() - t0 > 0.4
    t.join(5)
    assert result["out"][0] == "ok"                             # the load completed, nothing vanished mid-use
    assert not models.loaded() and models._timer is None

def test_idle_timer_never_reschedules_after_shutdown(fake_mlx, monkeypatch):
    monkeypatch.setattr(config, "RAG_IDLE_UNLOAD_S", 0.05)
    llm = models.shared_llm()
    with models.heavy("answer"):
        llm.generate("s", "u")
    models.shutdown()
    assert models._timer is None
    models._schedule()                                          # e.g. a timer callback racing with shutdown
    llm.unload()                                                # or a late caller
    assert models._timer is None
    time.sleep(0.3)
    assert models._timer is None

# ---------------------------------------------------------------- integration

def _hash_index(monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    jobs._export_state.update(pending=0, last=0.0)

def test_graph_extraction_runs_inside_the_heavy_lock(monkeypatch):
    from test_graph_sync import FakeLLM, FakeStore, triton_reel
    _hash_index(monkeypatch)
    triton_reel("A")
    indexer.sync_all()
    seen = []

    class Checking(FakeLLM):
        def generate(self, system, user, max_tokens=0):
            seen.append(models.held_by_me() and models.holder()["purpose"])
            return super().generate(system, user, max_tokens)
    try:
        assert gs.sync(store=FakeStore(), llm=Checking())["extracted"] == 1
    finally:
        indexer.release_writer()
    assert seen and set(seen) == {"graph extraction"}

def test_graph_sync_uses_the_shared_llm_when_none_is_given(monkeypatch, fake_mlx):
    from test_graph_sync import FakeStore, triton_reel
    monkeypatch.setattr(gs, "extractor_version", lambda model_id=None: "x-shared")
    monkeypatch.setattr(gs, "compatible_versions", lambda model_id=None: [])
    _hash_index(monkeypatch)
    triton_reel("A")
    indexer.sync_all()
    try:
        gs.sync(store=FakeStore())
    finally:
        indexer.release_writer()
    assert fake_mlx.instances == 1 and models._shared is not None  # the shared copy, not a private one

def test_index_embedding_runs_inside_the_heavy_lock(monkeypatch):
    from test_indexer import add
    _hash_index(monkeypatch)
    seen = []
    real = embedder.HashEmbedder.encode_docs

    def checking(self, texts):
        seen.append(models.holder()["purpose"] if models.held_by_me() else None)
        return real(self, texts)
    monkeypatch.setattr(embedder.HashEmbedder, "encode_docs", checking)
    add("A", "Triton is built on top of CUDA")
    try:
        indexer.sync_all()
    finally:
        indexer.release_writer()
    assert seen and set(seen) == {"index embedding"}

def test_background_graph_sync_yields_to_interactive_requests_and_to_capture(monkeypatch):
    monkeypatch.setattr(gs, "_stop", threading.Event())          # independent of other tests' gs.stop()
    assert gs._idle() is True
    with models.interactive():
        assert models.interactive_waiting() and gs._idle() is False       # stops at the next Reel boundary
    assert gs._idle() is True
    jobs.create_session("instagram-live", "instagram", 0)                  # capture in progress
    assert gs._idle() is False

# ---------------------------------------------------------------- Phase 4C: configurable answer model

def test_answer_model_follows_rag_llm_and_each_model_has_one_instance(fake_mlx, monkeypatch):
    monkeypatch.setattr(config, "RAG_LLM", "answer-model")
    a, g = models.shared_llm(), models.shared_llm(config.GRAPH_LLM)
    assert a.model_id == "answer-model" and g.model_id == config.GRAPH_LLM and a is not g
    assert models.shared_llm() is a and models.shared_llm(config.GRAPH_LLM) is g and fake_mlx.instances == 2

def test_only_one_model_is_resident_at_a_time(fake_mlx, monkeypatch):
    monkeypatch.setattr(config, "RAG_LLM", "answer-model")
    a, g = models.shared_llm(), models.shared_llm(config.GRAPH_LLM)
    with models.heavy("graph extraction"):
        g.generate("s", "u")
    assert models.resident() == [config.GRAPH_LLM]
    with models.heavy("answer generation"):
        a.generate("s", "u")                                    # frees the extraction model first
    assert models.resident() == ["answer-model"] and models.stats["evictions"] == 1
    with models.heavy("graph extraction"):
        g.generate("s", "u")
    assert models.resident() == [config.GRAPH_LLM] and a._inner.loads == 1 and g._inner.loads == 2

def test_idle_unload_and_shutdown_free_every_model(fake_mlx, monkeypatch):
    monkeypatch.setattr(config, "RAG_LLM", "answer-model")
    a = models.shared_llm()
    with models.heavy("answer"):
        a.generate("s", "u")
    models._last_use = time.monotonic() - 120
    assert models.unload_if_idle(idle_s=60) and models.resident() == []
    with models.heavy("answer"):
        a.generate("s", "u")
    models.shutdown()
    assert models.resident() == []

def test_graph_sync_keeps_using_the_extraction_model_when_answers_use_another(monkeypatch, fake_mlx):
    from test_graph_sync import FakeStore, triton_reel
    monkeypatch.setattr(config, "RAG_LLM", "answer-model")
    monkeypatch.setattr(gs, "extractor_version", lambda model_id=None: "x-shared")
    monkeypatch.setattr(gs, "compatible_versions", lambda model_id=None: [])
    _hash_index(monkeypatch)
    triton_reel("A")
    indexer.sync_all()
    try:
        gs.sync(store=FakeStore())
    finally:
        indexer.release_writer()
    assert list(models._llms) == [config.GRAPH_LLM]               # the answer model was never created