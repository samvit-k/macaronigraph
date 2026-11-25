"""Shared SQLite helpers: WAL connections, serialized write transactions and UTC timestamps."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def connect(path: Path, schema: str, *, foreign_keys: bool = False, migrate=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    if foreign_keys:
        con.execute("PRAGMA foreign_keys=ON")
    con.executescript(schema)
    if migrate:
        migrate(con)
    try:
        yield con
    finally:
        con.close()

@contextmanager
def transaction(lock, connect_fn):
    """BEGIN IMMEDIATE under the caller's process-wide lock; rolls back on any exception, including interrupts."""
    with lock, connect_fn() as con:
        con.execute("BEGIN IMMEDIATE")
        try:
            yield con
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise