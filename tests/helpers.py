"""Shared test helpers: free ports, the per-test production-path guard, table digests and a held heavy-work lock."""
from __future__ import annotations

import hashlib
import socket
import threading
from contextlib import contextmanager

from backend import config, indexer, jobs, models


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def production_path_guard(isolated_output):
    """Installed outside monkeypatch so a rollback cannot remove it: any database or Chroma path outside the
    temporary OUTPUT_DIR fails the test instead of touching production data."""
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


def canonical_digest() -> str:
    with jobs.connect() as con:
        h = hashlib.sha256()
        for t in ("reels", "attempts", "sessions"):
            for row in con.execute(f"SELECT * FROM {t} ORDER BY rowid"):
                h.update(repr(tuple(row)).encode())
        return h.hexdigest()


@contextmanager
def heavy_lock_held(label: str = "graph extraction"):
    """Another thread holds the heavy-work lock for the duration of the block."""
    entered, release = threading.Event(), threading.Event()

    def hold():
        with models.heavy(label):
            entered.set()
            release.wait(10)
    t = threading.Thread(target=hold)
    t.start()
    entered.wait(5)
    try:
        yield
    finally:
        release.set()
        t.join(5)
