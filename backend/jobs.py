"""SQLite job tracking: resumable, idempotent ingestion with retries and one canonical record per Reel.

CLI:  python -m backend.jobs status | requeue <reel_id>... | export | migrate [phase1.jsonl]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from . import config, db

MAX_ATTEMPTS = 3
PERMANENT_CONFIRM = 2
STALE_SESSION_S = 600
EXPORT_EVERY_N = 10
EXPORT_EVERY_S = 60
RANK = {"ok": 4, "ok_no_speech": 3, "partial": 2, "failed": 1}
ACCEPTABLE_PARTIAL = ("truncated_at_capture_limit", "metadata_not_found")
FREE_RETRY_PLAYBACK = {"auth_required", "aborted", "capture_lost"}

_lock = threading.RLock()
_export_state = {"pending": 0, "last": 0.0}
SCHEMA = """
CREATE TABLE IF NOT EXISTS reels (
  reel_id TEXT PRIMARY KEY, source TEXT NOT NULL, source_url TEXT, source_index INTEGER, media_type TEXT,
  saved_at TEXT, state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
  last_status TEXT, last_error TEXT, lease_session TEXT, canonical_attempt_id INTEGER, updated_at TEXT);
CREATE INDEX IF NOT EXISTS reels_state ON reels(source, state, source_index);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, source TEXT, mode TEXT, created_at REAL, last_seen REAL,
  finished_at REAL, end_reason TEXT, items TEXT);
CREATE TABLE IF NOT EXISTS attempts (
  attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, reel_id TEXT NOT NULL, session_id TEXT NOT NULL,
  processed_at TEXT, status TEXT, errors TEXT, counted INTEGER, outcome TEXT, record TEXT NOT NULL,
  UNIQUE(session_id, reel_id));
CREATE INDEX IF NOT EXISTS attempts_reel ON attempts(reel_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

_LINE_BREAKERS = {"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}

def jsonl_line(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    for raw, esc in _LINE_BREAKERS.items():
        s = s.replace(raw, esc)
    return s

def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL splitting on '\\n' only (never str.splitlines, which also splits on \\x85/U+2028)."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]

def db_path() -> Path:
    return config.OUTPUT_DIR / "pipeline.db"

def attempts_jsonl() -> Path:
    return config.OUTPUT_DIR / "attempts.jsonl"

def _migrate(con) -> None:
    if "prior_state" not in {r[1] for r in con.execute("PRAGMA table_info(reels)")}:
        con.execute("ALTER TABLE reels ADD COLUMN prior_state TEXT")
    if "capture_finished_at" not in {r[1] for r in con.execute("PRAGMA table_info(sessions)")}:
        con.execute("ALTER TABLE sessions ADD COLUMN capture_finished_at REAL")

def connect():
    return db.connect(db_path(), SCHEMA, migrate=_migrate)

def tx():
    return db.transaction(_lock, connect)

def get_meta(key: str) -> str | None:
    with connect() as con:
        row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None

def set_meta(key: str, value: str) -> None:
    with tx() as con:
        con.execute("INSERT INTO meta(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value))

def import_posts(source: str, posts: list[dict]) -> int:
    with tx() as con:
        before = con.execute("SELECT COUNT(*) FROM reels").fetchone()[0]
        con.executemany(
            """INSERT INTO reels(reel_id, source, source_url, source_index, media_type, saved_at, state, updated_at)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(reel_id) DO UPDATE SET source_url=excluded.source_url,
                 source_index=excluded.source_index, saved_at=excluded.saved_at, media_type=excluded.media_type""",
            [(p["reel_id"], source, p["source_url"], p["source_index"], p["media_type"], p["saved_at"],
              "pending" if p["media_type"] == "reel" else "skipped", db.utc_now()) for p in posts if p["reel_id"]])
        return con.execute("SELECT COUNT(*) FROM reels").fetchone()[0] - before

def _release(con, where: str, args: tuple) -> int:
    return con.execute(
        f"""UPDATE reels SET state = COALESCE(prior_state,
                                     CASE WHEN attempts > 0 THEN 'needs_retry' ELSE 'pending' END),
              prior_state = NULL, lease_session = NULL, updated_at = ?
            WHERE state = 'in_progress' AND {where}""",
        (db.utc_now(), *args)).rowcount

def reclaim_stale(con, now: float | None = None) -> int:
    """A finished session whose uploads are still queued is draining, not dead, and is never abandoned."""
    now = now or time.time()
    con.execute("""UPDATE sessions SET finished_at = ?, end_reason = 'abandoned (no heartbeat)'
                   WHERE finished_at IS NULL AND capture_finished_at IS NULL AND last_seen < ?""",
                (now, now - STALE_SESSION_S))
    return _release(con, """(lease_session IS NULL OR lease_session IN
                             (SELECT session_id FROM sessions WHERE finished_at IS NOT NULL))""", ())

def create_session(session_id: str, source: str, limit: int, offset: int = 0, mode: str = "pending",
                   now: float | None = None, reel_ids: list[str] | None = None) -> list[dict]:
    states = ("pending", "needs_retry") if mode == "pending" else \
             ("pending", "needs_retry", "done", "exhausted", "failed_permanent")
    now = now or time.time()
    with tx() as con:
        reclaim_stale(con, now)
        reel_ids = list(reel_ids or [])
        id_filter = f" AND reel_id IN ({','.join('?' * len(reel_ids))})" if mode == "eval" else ""
        rows = [] if mode == "eval" and not reel_ids else con.execute(
            f"""SELECT reel_id, source_url, source_index, state, attempts FROM reels
                WHERE source = ? AND state IN ({",".join("?" * len(states))}){id_filter}
                ORDER BY source_index LIMIT ? OFFSET ?""",
            (source, *states, *(reel_ids if mode == "eval" else []), limit, offset)).fetchall()
        items = [{"reel_id": r["reel_id"], "url": r["source_url"], "index": r["source_index"],
                  "prior_state": r["state"], "prior_attempts": r["attempts"]} for r in rows]
        con.execute("INSERT INTO sessions(session_id, source, mode, created_at, last_seen, items) VALUES(?,?,?,?,?,?)",
                    (session_id, source, mode, now, now, json.dumps([i["reel_id"] for i in items])))
        con.executemany("UPDATE reels SET prior_state=state, state='in_progress', lease_session=?, updated_at=? "
                        "WHERE reel_id=?", [(session_id, db.utc_now(), i["reel_id"]) for i in items])
    return items

def heartbeat(session_id: str, now: float | None = None) -> bool:
    with tx() as con:
        return con.execute("UPDATE sessions SET last_seen = ? WHERE session_id = ? AND finished_at IS NULL",
                           (now or time.time(), session_id)).rowcount == 1

def finish_session(session_id: str, reason: str = "finished") -> int:
    with tx() as con:
        con.execute("UPDATE sessions SET capture_finished_at = COALESCE(capture_finished_at, ?), "
                    "end_reason = COALESCE(end_reason, ?) WHERE session_id = ?", (time.time(), reason, session_id))
        released = _release(con, "lease_session = ?", (session_id,))
        _maybe_finalize(con, session_id)
    export_canonical()
    return released

def _maybe_finalize(con, session_id: str | None) -> None:
    if session_id and not con.execute("SELECT 1 FROM reels WHERE state='queued' AND lease_session=?",
                                      (session_id,)).fetchone():
        con.execute("UPDATE sessions SET finished_at = COALESCE(finished_at, ?) "
                    "WHERE session_id = ? AND capture_finished_at IS NOT NULL", (time.time(), session_id))

def session_progress(session_id: str) -> dict:
    with connect() as con:
        s = con.execute("SELECT capture_finished_at, finished_at, end_reason FROM sessions WHERE session_id=?",
                        (session_id,)).fetchone()
        queued = con.execute("SELECT COUNT(*) FROM reels WHERE state='queued' AND lease_session=?",
                             (session_id,)).fetchone()[0]
    return {"session_id": session_id, "known": s is not None, "queued": queued,
            "capture_finished": bool(s and s["capture_finished_at"]), "finalized": bool(s and s["finished_at"]),
            "end_reason": s["end_reason"] if s else None}

def mark_queued(reel_id: str, session_id: str) -> dict | None:
    """Called before the job is published, so the worker can never record the attempt first and have this overwrite it."""
    with tx() as con:
        row = con.execute("SELECT state, lease_session, prior_state FROM reels WHERE reel_id=?", (reel_id,)).fetchone()
        con.execute("UPDATE reels SET prior_state = CASE WHEN state IN ('in_progress','queued') THEN prior_state "
                    "ELSE state END, state='queued', lease_session=?, updated_at=? WHERE reel_id=?",
                    (session_id, db.utc_now(), reel_id))
    return dict(row) if row else None

def revert_queued(reel_id: str, snapshot: dict | None) -> None:
    if snapshot is None:
        return
    with tx() as con:
        con.execute("UPDATE reels SET state=?, lease_session=?, prior_state=?, updated_at=? "
                    "WHERE reel_id=? AND state='queued'",
                    (snapshot["state"], snapshot["lease_session"], snapshot["prior_state"], db.utc_now(), reel_id))

def restore_orphan_queued(keep: list[str]) -> int:
    with tx() as con:
        ph = ",".join("?" * len(keep))
        return con.execute(
            f"""UPDATE reels SET state = COALESCE(prior_state,
                                     CASE WHEN attempts > 0 THEN 'needs_retry' ELSE 'pending' END),
                  prior_state = NULL, lease_session = NULL, updated_at = ?
                WHERE state = 'queued' {f'AND reel_id NOT IN ({ph})' if keep else ''}""",
            (db.utc_now(), *keep)).rowcount

def classify(record: dict) -> str:
    status, errors = record["status"], record.get("errors", [])
    if status in ("ok", "ok_no_speech"):
        return "done"
    if status == "partial":
        return "done" if errors and all(e.startswith(ACCEPTABLE_PARTIAL) for e in errors) else "retry"
    cap = record.get("capture") or {}
    if cap.get("playback") in FREE_RETRY_PLAYBACK:
        return "retry_free"
    if cap.get("playback") == "unavailable" and "rate limit" not in str(cap.get("playback_detail") or "").lower():
        return "permanent"
    return "retry"

def _score(rec: dict) -> tuple:
    t, o = rec.get("transcript") or {}, rec.get("ocr") or {}
    return (RANK.get(rec["status"], 0), len(t.get("text", "")) + sum(len(s["text"]) for s in o.get("segments", [])))

def existing_attempt(session_id: str | None, reel_id: str | None) -> dict | None:
    if not session_id or not reel_id:
        return None
    with connect() as con:
        row = con.execute("SELECT attempt_id, status, errors, outcome FROM attempts WHERE session_id=? AND reel_id=?",
                          (session_id, reel_id)).fetchone()
    return dict(row) if row else None

def record_attempt(record: dict, export: bool = True) -> dict:
    rid = record["reel_id"]
    sid = record.get("session_id") or f"nosession-{record.get('processed_at')}"
    outcome = classify(record)
    counted = outcome in ("retry", "permanent", "done")
    with tx() as con:
        dup = con.execute("SELECT attempt_id, outcome FROM attempts WHERE session_id=? AND reel_id=?",
                          (sid, rid)).fetchone()
        if dup:
            # Already persisted (crash after commit, then re-run): change nothing.
            con.execute("""UPDATE reels SET state = COALESCE(prior_state, state), prior_state = NULL,
                             lease_session = NULL, updated_at = ?
                           WHERE reel_id = ? AND state = 'queued' AND lease_session = ?""", (db.utc_now(), rid, sid))
            _maybe_finalize(con, sid)
            reel = con.execute("SELECT state FROM reels WHERE reel_id=?", (rid,)).fetchone()
            return {"attempt_id": dup["attempt_id"], "duplicate": True, "outcome": dup["outcome"],
                    "state": reel["state"] if reel else None}
        con.execute("UPDATE sessions SET last_seen = ? WHERE session_id = ?", (time.time(), sid))
        attempt_id = con.execute(
            "INSERT INTO attempts(reel_id, session_id, processed_at, status, errors, counted, outcome, record) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (rid, sid, record["processed_at"], record["status"], json.dumps(record["errors"]), int(counted), outcome,
             json.dumps(record, ensure_ascii=False))).lastrowid
        reel = con.execute("SELECT * FROM reels WHERE reel_id=?", (rid,)).fetchone()
        if reel is None:
            con.execute("INSERT INTO reels(reel_id, source, source_url, state, updated_at) VALUES(?,?,?,?,?)",
                        (rid, sid.split("-")[0], record.get("source_url"), "pending", db.utc_now()))
            reel = con.execute("SELECT * FROM reels WHERE reel_id=?", (rid,)).fetchone()

        attempts = reel["attempts"] + int(counted)
        if outcome == "done":
            state = "done"
        elif outcome == "retry_free":
            state = "needs_retry" if attempts else "pending"
        elif outcome == "permanent" and _count_outcome(con, rid, "permanent") >= PERMANENT_CONFIRM:
            state = "failed_permanent"
        else:
            state = "exhausted" if attempts >= MAX_ATTEMPTS else "needs_retry"
        if reel["state"] == "done" or (reel["state"] in ("in_progress", "queued") and _has_done(con, rid, attempt_id)):
            state = "done"  # a good canonical record never regresses because of a worse re-run

        canonical = _pick_canonical(con, rid)
        con.execute("""UPDATE reels SET state=?, attempts=?, last_status=?, last_error=?, lease_session=NULL,
                       prior_state=NULL, canonical_attempt_id=?, updated_at=? WHERE reel_id=?""",
                    (state, attempts, record["status"], "; ".join(record["errors"])[:500] or None,
                     canonical, db.utc_now(), rid))
        _maybe_finalize(con, sid)
    _append_history(record, attempt_id, outcome)
    if export:
        maybe_export()
    return {"attempt_id": attempt_id, "duplicate": False, "outcome": outcome, "state": state,
            "attempts": attempts, "canonical_attempt_id": canonical}

def _count_outcome(con, rid: str, outcome: str) -> int:
    return con.execute("SELECT COUNT(*) FROM attempts WHERE reel_id=? AND outcome=?", (rid, outcome)).fetchone()[0]

def _has_done(con, rid: str, exclude_id: int) -> bool:
    return con.execute("SELECT 1 FROM attempts WHERE reel_id=? AND outcome='done' AND attempt_id<>?",
                       (rid, exclude_id)).fetchone() is not None

def _pick_canonical(con, rid: str) -> int | None:
    best, best_key = None, None
    for row in con.execute("SELECT attempt_id, record FROM attempts WHERE reel_id=? ORDER BY attempt_id", (rid,)):
        key = (*_score(json.loads(row["record"])), row["attempt_id"])
        if best_key is None or key > best_key:
            best, best_key = row["attempt_id"], key
    return best

def _append_history(record: dict, attempt_id: int, outcome: str) -> None:
    with _lock, open(attempts_jsonl(), "a", encoding="utf-8") as f:
        f.write(jsonl_line({**record, "attempt_id": attempt_id, "outcome": outcome}) + "\n")

def maybe_export(force: bool = False) -> bool:
    with _lock:
        _export_state["pending"] += 1
        due = force or _export_state["pending"] >= EXPORT_EVERY_N or \
            time.time() - _export_state["last"] >= EXPORT_EVERY_S
    if due:
        export_canonical()
    return due

def export_canonical() -> int:
    """Atomic rewrite (temporary file, fsync, rename): readers never see a partial file."""
    with _lock, connect() as con:
        tmp = config.OUTPUT_JSONL.with_suffix(".jsonl.tmp")
        n = 0
        with open(tmp, "w", encoding="utf-8") as f:
            for r in con.execute("""SELECT r.reel_id, r.state, r.attempts, r.canonical_attempt_id, a.record
                                    FROM reels r JOIN attempts a ON a.attempt_id = r.canonical_attempt_id
                                    ORDER BY r.source, r.source_index"""):
                rec = json.loads(r["record"])
                rec["processing"] = {
                    "state": r["state"], "attempts_counted": r["attempts"],
                    "canonical_attempt_id": r["canonical_attempt_id"],
                    "history": [{"attempt_id": h["attempt_id"], "session_id": h["session_id"],
                                 "processed_at": h["processed_at"], "status": h["status"], "outcome": h["outcome"],
                                 "errors": json.loads(h["errors"] or "[]")}
                                for h in con.execute("SELECT attempt_id, session_id, processed_at, status, outcome, "
                                                     "errors FROM attempts WHERE reel_id=? ORDER BY attempt_id",
                                                     (r["reel_id"],))]}
                f.write(jsonl_line(rec) + "\n")
                n += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, config.OUTPUT_JSONL)
        _export_state.update(pending=0, last=time.time())
    return n

def status(source: str | None = None) -> dict:
    with connect() as con:
        out: dict = {}
        q = "SELECT source, state, COUNT(*) n FROM reels" + (" WHERE source=?" if source else "") + " GROUP BY 1, 2"
        for r in con.execute(q, (source,) if source else ()):
            out.setdefault(r["source"], {})[r["state"]] = r["n"]
        open_sessions = [dict(r) for r in con.execute(
            "SELECT session_id, created_at, last_seen FROM sessions WHERE finished_at IS NULL")]
        attempts = con.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    return {"reels": out, "attempts": attempts, "open_sessions": open_sessions}

def requeue(reel_ids: list[str], reset_attempts: bool = True) -> int:
    with tx() as con:
        return sum(con.execute(f"UPDATE reels SET state='pending', lease_session=NULL"
                               f"{', attempts=0' if reset_attempts else ''}, updated_at=? WHERE reel_id=?",
                               (db.utc_now(), rid)).rowcount for rid in reel_ids)

def migrate_phase1(path: Path | None = None, sources: dict[str, list[dict]] | None = None) -> dict:
    """Safe and resumable: the source file is backed up first, and an interrupted migration resumes from the backup."""
    explicit = path is not None and Path(path).resolve() != config.OUTPUT_JSONL.resolve()
    path = Path(path or config.OUTPUT_JSONL)
    if not explicit:
        marker = get_meta("phase1_migration_in_progress") or get_meta("phase1_migration")
        if marker:
            path = Path(json.loads(marker)["backup"])
            if not path.exists():
                raise FileNotFoundError(f"migration backup missing: {path}")
    if not path.exists():
        raise FileNotFoundError(path)

    backup_dir = config.OUTPUT_DIR / "backup"
    if path.parent.resolve() == backup_dir.resolve():
        backup = path
    else:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{path.stem}.phase1.{digest}.jsonl"
        if not backup.exists():
            tmp = backup.with_suffix(".tmp")
            shutil.copyfile(path, tmp)
            os.replace(tmp, backup)
        set_meta("phase1_migration_in_progress", json.dumps({"backup": str(backup)}))

    for source, posts in (sources or {}).items():
        import_posts(source, posts)
    imported = skipped = 0
    for line in backup.read_text(encoding="utf-8").split("\n"):  # never splitlines(): see read_jsonl
        if not line.strip():
            continue
        rec = json.loads(line)
        if "processing" in rec:
            skipped += 1
            continue
        if record_attempt(rec, export=False)["duplicate"]:
            skipped += 1
        else:
            imported += 1
    export_canonical()
    with tx() as con:
        con.execute("INSERT INTO meta(key, value) VALUES('phase1_migration', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (json.dumps({"backup": str(backup), "imported": imported, "at": db.utc_now()}),))
        con.execute("DELETE FROM meta WHERE key='phase1_migration_in_progress'")
    return {"backup": str(backup), "imported": imported, "skipped": skipped}

if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "status"), sys.argv[2:]
    if cmd == "status":
        print(json.dumps(status(), indent=2))
    elif cmd == "requeue":
        print(f"requeued {requeue(args)}")
    elif cmd == "export":
        print(f"exported {export_canonical()} canonical records → {config.OUTPUT_JSONL}")
    elif cmd == "migrate":
        from .app import _posts

        srcs = {}
        for s in ("test", "instagram"):
            try:
                srcs[s] = list(_posts(s).values())
            except Exception as e:  # noqa: BLE001
                print(f"skip source {s}: {e}", file=sys.stderr)
        print(json.dumps(migrate_phase1(Path(args[0]) if args else None, srcs), indent=2))
    else:
        sys.exit(__doc__)