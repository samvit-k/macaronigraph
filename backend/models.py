"""Phase 4B: coordination of expensive local model work.

heavy(purpose)   ONE expensive model operation at a time — answer generation, follow-up rewriting, graph
                 extraction, index embedding — across threads AND processes (thread lock + flock on
                 OUTPUT_DIR/.heavy-work.lock), so a CLI graph sync never runs its LLM while the backend embeds or
                 answers. Re-entrant within a thread. Capture, Whisper and OCR are not changed: model work
                 defers to them through indexer.busy_reason().
shared_llm(id)   the in-process LLM for a model id (graph.llm.MlxLLM behind a guard): config.RAG_LLM for answers
                 and rewriting (default), config.GRAPH_LLM for graph extraction (by default the same model). One
                 instance per model id, created once and never replaced — idle unloading only frees its weights,
                 which reload lazily — so two copies of a model never coexist. ONE MODEL RESIDENT AT A TIME: before
                 a model generates (heavy lock held, so nothing else is generating) the weights of every other
                 model are freed. Its generate() refuses to run unless the calling thread holds the heavy lock,
                 and unloading takes the heavy lock (without waiting), so unloading can never overlap a generation.
interactive()    a user-facing request marks itself as waiting; background graph syncs yield between Reels.
"""
from __future__ import annotations

import fcntl
import threading
import time
from contextlib import contextmanager

from . import config


class HeavyBusy(RuntimeError):
    """The heavy-work lock could not be acquired within the timeout."""


_tlock = threading.Lock()
_owner: dict = {"thread": None, "depth": 0, "purpose": None, "since": None}
_interactive = 0
_interactive_lock = threading.Lock()


def _lock_path():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return config.OUTPUT_DIR / ".heavy-work.lock"


def held_by_me() -> bool:
    return _owner["thread"] == threading.get_ident()


@contextmanager
def heavy(purpose: str, timeout: float | None = None):
    """Exclusive expensive-model section. timeout=None waits as long as needed; otherwise HeavyBusy.
    Only lock contention is retried; any other filesystem error propagates (with every lock released)."""
    if held_by_me():                                            # re-entrant (e.g. nested helpers)
        _owner["depth"] += 1
        try:
            yield
        finally:
            _owner["depth"] -= 1
        return
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    if not _tlock.acquire(timeout=-1 if timeout is None else max(0.0, timeout)):
        raise HeavyBusy(f"{_owner['purpose'] or 'another operation'} is using the local model")
    fd, locked = None, False
    try:
        fd = open(_lock_path(), "a+")
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:                             # contention only (EWOULDBLOCK / EAGAIN)
                if deadline is not None and time.monotonic() >= deadline:
                    raise HeavyBusy("another process is using the local model (e.g. a CLI graph sync)")
                time.sleep(0.1)
        _owner.update(thread=threading.get_ident(), depth=1, purpose=purpose, since=time.time())
        try:
            yield
        finally:
            _owner.update(thread=None, depth=0, purpose=None, since=None)
            _touch()
    finally:
        try:
            if fd is not None:
                try:
                    if locked:                                  # only unlock what was actually locked
                        fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    fd.close()
        finally:
            _tlock.release()                                    # always, even if file cleanup failed


def holder() -> dict | None:
    """What holds the heavy lock in this process (None if free here; another process may still hold it)."""
    return None if _owner["thread"] is None else {"purpose": _owner["purpose"], "since": _owner["since"]}


@contextmanager
def interactive():
    """Mark a user-facing request as waiting for or using the model (background syncs yield to it)."""
    global _interactive
    with _interactive_lock:
        _interactive += 1
    try:
        yield
    finally:
        with _interactive_lock:
            _interactive -= 1


def interactive_waiting() -> bool:
    return _interactive > 0


# ---------------------------------------------------------------- the shared LLM

class _SharedLLM:
    """Guard around the one MlxLLM instance: generation only inside heavy(); unload is deferred to the idle
    timer (callers such as graph sync release the model instead of dropping it)."""

    def __init__(self, inner, model_id: str | None = None):
        self._inner = inner
        self.model_id = model_id

    @property
    def load_seconds(self) -> float:
        return getattr(self._inner, "load_seconds", 0.0)

    def generate(self, system: str, user: str, max_tokens: int = 2048):
        if not held_by_me():
            raise RuntimeError("shared LLM used outside models.heavy(): refusing to generate")
        _evict_except(self)                                     # one model resident at a time
        try:
            return self._inner.generate(system, user, max_tokens=max_tokens)
        finally:
            _touch()

    def unload(self) -> None:
        _schedule()                                             # freed by the idle timer, never mid-use

    @property
    def weights_loaded(self) -> bool:
        return getattr(self._inner, "_model", None) is not None


_shared: _SharedLLM | None = None          # the most recently requested model
_llms: dict = {}                           # model id → its single wrapper
_create_lock = threading.Lock()
_last_use = 0.0
_timer: threading.Timer | None = None
_timer_lock = threading.Lock()
_closed = False                    # set by shutdown(): the idle timer never reschedules afterwards
stats = {"created": 0, "unloads": 0, "evictions": 0}


def _wrappers() -> list:
    ws = list(_llms.values())
    if _shared is not None and _shared not in ws:
        ws.append(_shared)
    return ws


def _evict_except(keep) -> None:
    """Free the weights of every other model (caller holds the heavy lock: none of them is generating)."""
    for w in _wrappers():
        if w is not keep and w.weights_loaded:
            w._inner.unload()
            stats["evictions"] += 1


def _touch() -> None:
    global _last_use
    _last_use = time.monotonic()


def shared_llm(model_id: str | None = None) -> _SharedLLM:
    """The in-process LLM for model_id (default config.RAG_LLM; weights load lazily inside the first generate)."""
    global _shared
    mid = model_id or config.RAG_LLM
    with _create_lock:
        w = _llms.get(mid)
        if w is None:
            from .graph.llm import MlxLLM
            w = _llms[mid] = _SharedLLM(MlxLLM(mid), mid)
            stats["created"] += 1
        _shared = w
    _touch()
    _schedule()
    return _shared


def loaded() -> bool:
    return any(w.weights_loaded for w in _wrappers())


def resident() -> list[str]:
    """Model ids whose weights are currently in memory (at most one after any generation)."""
    return [w.model_id for w in _wrappers() if w.weights_loaded]


def unload_if_idle(idle_s: float | None = None) -> bool:
    """Free the shared LLM's weights if unused for idle_s. Takes the heavy lock without waiting, so it never
    overlaps a generation in this process. → unloaded?"""
    idle_s = config.RAG_IDLE_UNLOAD_S if idle_s is None else idle_s
    if not loaded() or time.monotonic() - _last_use < idle_s:
        return False
    try:
        with heavy("idle unload", timeout=0):
            if time.monotonic() - _last_use < idle_s:           # used while we were acquiring
                return False
            for w in _wrappers():
                if w.weights_loaded:
                    w._inner.unload()
                    stats["unloads"] += 1
            return True
    except HeavyBusy:
        return False


def _schedule() -> None:
    global _timer

    def check():
        if not unload_if_idle() and loaded():
            _schedule()

    with _timer_lock:
        if _closed:
            return
        if _timer is not None:
            _timer.cancel()
        _timer = threading.Timer(max(1.0, config.RAG_IDLE_UNLOAD_S), check)
        _timer.daemon = True
        _timer.start()


def shutdown() -> None:
    """Backend shutdown: stop the idle timer for good, then free the weights. Waits for a running generation —
    including one still loading the weights — so nothing is unloaded mid-use."""
    global _timer, _closed
    with _timer_lock:
        _closed = True
        if _timer is not None:
            _timer.cancel()
            _timer = None
    if _wrappers():
        with heavy("shutdown"):
            for w in _wrappers():
                if w.weights_loaded:
                    w._inner.unload()
