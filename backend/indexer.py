"""Phase 3: incremental semantic index (ChromaDB) over the canonical Reel records in SQLite.

SQLite stays the source of truth. Chroma is a derived cache that can always be rebuilt from it.
`index_state` (same pipeline.db, created additively) records what each collection holds per Reel:
  (collection, reel_id) → canonical_attempt_id, content_hash, chunk_ids, status (indexed | empty | error)

Sync (idempotent, crash-safe, resumable):
  eligible Reels = canonical record with status ok / ok_no_speech / partial
  unchanged canonical_attempt_id (attempt rows are immutable) → skipped without parsing or embedding
  otherwise: chunk → hash; same hash → bookkeeping only; else embed → upsert → delete the Reel's obsolete
  ids (queried from Chroma itself) → commit the index_state row LAST. A crash anywhere just redoes that Reel.
  When the idle check fails mid-sync, buffered (not yet embedded) work is dropped uncommitted and redone
  on the next sync; committed Reels stay committed.

Rebuilds (embedding spec / chunker changed, or forced) go into a NEW collection while the active one keeps
serving and keeps being updated. The new one becomes active only after catch-up with SQLite (repeated
until a pass changes nothing) and a full verification pass; on failure the previous index stays active.
The replaced collection is retired, then dropped once no search uses it and a grace period has passed.

Automatic indexing runs only when capture and processing are idle; manual sync is always allowed.
One writer: an in-process lock + an flock on output/chroma/.writer.lock (the backend holds it while running).

CLI:  python -m backend.indexer status | sync | rebuild | verify | search "query" [k]
"""
from __future__ import annotations

import fcntl
import json
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone

from . import chunking, config, embedder, jobs, worker

ELIGIBLE = ("ok", "ok_no_speech", "partial")
EMBED_BATCH_CHUNKS = 64       # chunks embedded per model call (spanning several Reels)
CATCHUP_MAX_PASSES = 5
VERIFY_ROUNDS = 3             # catch-up + verify attempts before a rebuild is reported failed_verification
POLL_S = 60
STOP_JOIN_S = 30
RETIRE_GRACE_S = 300          # a replaced collection outlives the switch at least this long (other processes)
TYPE_GROUPS = {"caption": ["caption"], "transcript": ["transcript"],
               "ocr": ["ocr", "ocr_title", "ocr_fragment"], "ocr_title": ["ocr_title"],
               "ocr_fragment": ["ocr_fragment"], "ocr_screen": ["ocr"]}
SCHEMA = """
CREATE TABLE IF NOT EXISTS index_state (
  collection TEXT NOT NULL, reel_id TEXT NOT NULL, canonical_attempt_id INTEGER, content_hash TEXT,
  chunk_ids TEXT, n_chunks INTEGER, status TEXT NOT NULL, error TEXT, indexed_at TEXT,
  PRIMARY KEY (collection, reel_id));
"""
K_ACTIVE, K_BUILDING, K_BUILD, K_LAST, K_RETIRED = (
    "index.active", "index.building", "index.build_status", "index.last_sync", "index.retired")


class IndexNotReady(RuntimeError):
    pass


class WriterBusy(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _db():
    with jobs.connect() as con:
        con.executescript(SCHEMA)
        yield con


def _ensure_schema() -> None:
    with _db():
        pass


# ---------------------------------------------------------------- writer lock

_fd_lock = threading.Lock()        # guards _lock_fd and the client cache
_write_lock = threading.RLock()    # held for the whole duration of any index write
_lock_fd = None


def acquire_writer() -> bool:
    """Take the cross-process writer lock (non-blocking). True if this process holds it."""
    global _lock_fd
    with _fd_lock:
        if _lock_fd is not None:
            return True
        d = config.chroma_dir()
        d.mkdir(parents=True, exist_ok=True)
        fd = open(d / ".writer.lock", "a+")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fd.close()
            return False
        _lock_fd = fd
        return True


def release_writer() -> bool:
    """Release the cross-process lock — never while another thread is inside a write. → released?"""
    global _lock_fd
    if not _write_lock.acquire(blocking=False):
        return False
    try:
        with _fd_lock:
            if _lock_fd is not None:
                fcntl.flock(_lock_fd, fcntl.LOCK_UN)
                _lock_fd.close()
                _lock_fd = None
        return True
    finally:
        _write_lock.release()


def holds_writer() -> bool:
    return _lock_fd is not None


@contextmanager
def writing():
    """Serialize index writes in-process and across processes."""
    with _write_lock:
        held = _lock_fd is not None
        if not held and not acquire_writer():
            raise WriterBusy("another process (the backend?) holds the index writer lock")
        try:
            yield
        finally:
            if not held:
                release_writer()


# ---------------------------------------------------------------- chroma

_clients: dict[str, object] = {}


def client():
    import chromadb
    from chromadb.config import Settings

    path = str(config.chroma_dir())
    with _fd_lock:
        if path not in _clients:
            config.chroma_dir().mkdir(parents=True, exist_ok=True)
            _clients[path] = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        return _clients[path]


def collections() -> list[str]:
    return [c.name for c in client().list_collections()]


def get_collection(name: str):
    return client().get_collection(name, embedding_function=None)


def spec_of(col) -> dict:
    return json.loads(col.metadata["spec"])


def _create_collection(spec: dict) -> str:
    name = f"reels_{embedder.signature(spec)}_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"
    client().create_collection(name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
                               metadata={"spec": json.dumps(spec, sort_keys=True),
                                         "signature": embedder.signature(spec), "created_at": _now()})
    return name


def _drop_collection(name: str) -> None:
    if name in collections():
        client().delete_collection(name)
    with jobs.tx() as con:
        con.execute("DELETE FROM index_state WHERE collection=?", (name,))


def active_collection() -> str | None:
    name = jobs.get_meta(K_ACTIVE)
    return name if name and name in collections() else None


def _meta_json(key: str):
    v = jobs.get_meta(key)
    return json.loads(v) if v else None


# ---------------------------------------------------------------- readers + retirement of replaced collections

_readers: Counter = Counter()
_readers_lock = threading.Lock()


@contextmanager
def reading():
    """Pin the active collection for one search: it cannot be dropped while pinned."""
    with _readers_lock:
        name = active_collection()
        if name:
            _readers[name] += 1
    try:
        yield name
    finally:
        if name:
            with _readers_lock:
                _readers[name] -= 1


def _retire(name: str) -> None:
    retired = [r for r in (_meta_json(K_RETIRED) or []) if r["name"] != name]
    jobs.set_meta(K_RETIRED, json.dumps(retired + [{"name": name, "at": time.time()}]))


def cleanup_retired(now: float | None = None) -> list[str]:
    """Drop retired collections that are past the grace period and not in use by any search here."""
    now = now or time.time()
    retired = _meta_json(K_RETIRED) or []
    keep, dropped = [], []
    with writing():
        for r in retired:
            with _readers_lock:                       # no new reader can pin it while we decide and drop
                if r["name"] == jobs.get_meta(K_ACTIVE):
                    continue                          # re-activated: not retired any more
                if _readers[r["name"]] == 0 and now - r["at"] >= RETIRE_GRACE_S:
                    _drop_collection(r["name"])
                    dropped.append(r["name"])
                    continue
            keep.append(r)
        jobs.set_meta(K_RETIRED, json.dumps(keep))
    return dropped


# ---------------------------------------------------------------- SQLite side (read-only for jobs tables)

def eligible() -> dict[str, dict]:
    """Reels whose canonical record should be searchable → {reel_id: {source, aid, status}}."""
    with _db() as con:
        rows = con.execute("""SELECT r.reel_id, r.source, r.canonical_attempt_id AS aid, a.status
                              FROM reels r JOIN attempts a ON a.attempt_id = r.canonical_attempt_id""").fetchall()
    return {r["reel_id"]: {"source": r["source"], "aid": r["aid"], "status": r["status"]}
            for r in rows if r["status"] in ELIGIBLE}


def load_record(attempt_id: int) -> dict:
    with _db() as con:
        row = con.execute("SELECT record FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
    if row is None:
        raise KeyError(f"attempt {attempt_id} not found")
    return json.loads(row["record"])


def state_rows(collection: str) -> dict[str, dict]:
    with _db() as con:
        return {r["reel_id"]: dict(r) for r in
                con.execute("SELECT * FROM index_state WHERE collection=?", (collection,))}


def _write_state(collection: str, rid: str, aid: int | None, h: str | None, ids: list[str], status: str,
                 error: str | None = None) -> None:
    with jobs.tx() as con:
        con.execute("""INSERT INTO index_state(collection, reel_id, canonical_attempt_id, content_hash, chunk_ids,
                         n_chunks, status, error, indexed_at) VALUES(?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(collection, reel_id) DO UPDATE SET canonical_attempt_id=excluded.canonical_attempt_id,
                         content_hash=excluded.content_hash, chunk_ids=excluded.chunk_ids, n_chunks=excluded.n_chunks,
                         status=excluded.status, error=excluded.error, indexed_at=excluded.indexed_at""",
                    (collection, rid, aid, h, json.dumps(ids), len(ids), status, error, _now()))


def _write_error(collection: str, rid: str, aid: int, prev: dict | None, msg: str) -> None:
    """Record a failed (re)index WITHOUT forgetting the last successful one: the row keeps the previously
    committed attempt id, content hash and chunk_ids, so those chunks stay protected and searchable. The row's
    `error` status makes the next sync retry it, and verification refuses to activate a collection with it."""
    if prev and prev.get("status") in ("indexed", "empty", "error") and prev.get("chunk_ids"):
        _write_state(collection, rid, prev["canonical_attempt_id"], prev["content_hash"],
                     json.loads(prev["chunk_ids"]), "error", f"attempt {aid}: {msg}")
    else:
        _write_state(collection, rid, aid, None, [], "error", f"attempt {aid}: {msg}")


def _delete_state(collection: str, rids: list[str]) -> None:
    if rids:
        with jobs.tx() as con:
            con.executemany("DELETE FROM index_state WHERE collection=? AND reel_id=?", [(collection, r) for r in rids])


# ---------------------------------------------------------------- sync

def _apply(col, collection: str, rid: str, aid: int, chunks: list[dict], vecs, h: str) -> None:
    """Make Chroma hold exactly `chunks` for this Reel, then record it (bookkeeping last)."""
    ids = [c["id"] for c in chunks]
    if chunks:
        col.upsert(ids=ids, embeddings=[v.tolist() for v in vecs], documents=[c["text"] for c in chunks],
                   metadatas=[{**c["metadata"], "content_hash": h} for c in chunks])
    obsolete = set(col.get(where={"reel_id": rid}, include=[])["ids"]) - set(ids)
    if obsolete:
        col.delete(ids=sorted(obsolete))
    _write_state(collection, rid, aid, h, ids, "indexed" if chunks else "empty")


def _remove_reel(col, collection: str, rid: str) -> None:
    ids = col.get(where={"reel_id": rid}, include=[])["ids"]
    if ids:
        col.delete(ids=ids)
    _delete_state(collection, [rid])


def sweep_orphans(col, collection: str, elig: dict) -> int:
    """Delete every chunk the bookkeeping does not expect: chunks of ineligible / unrecorded Reels, and chunks
    whose id is not in their Reel's recorded chunk_ids. Error rows keep the last successful chunk_ids
    (_write_error), so a failed re-index never deletes previously valid embeddings. Runs after every
    completed sync, also when nothing needed re-embedding. → number deleted"""
    rows = state_rows(collection)
    expected = {rid: set(json.loads(r["chunk_ids"] or "[]")) for rid, r in rows.items()}
    got = col.get(include=["metadatas"])
    orphans = [cid for cid, m in zip(got["ids"], got["metadatas"])
               if (rid := (m or {}).get("reel_id")) not in elig or rid not in rows or cid not in expected[rid]]
    if orphans:
        col.delete(ids=orphans)
    return len(orphans)


def sync_collection(collection: str, should_continue=None) -> dict:
    """Bring one collection up to date with SQLite (with its own embedding model). When `should_continue()`
    turns False, stop before the next embedding call: buffered work is dropped uncommitted (redone next
    time), committed Reels stay."""
    t0 = time.time()
    col = get_collection(collection)
    emb = embedder.for_spec(spec_of(col))
    elig = eligible()
    rows = state_rows(collection)
    st = {"collection": collection, "eligible": len(elig), "skipped": 0, "embedded": 0, "chunks_written": 0,
          "unchanged_hash": 0, "removed": 0, "orphans_deleted": 0, "errors": 0, "stopped": False, "deferred": 0}

    def go_on() -> bool:
        if should_continue is not None and not should_continue():
            st["stopped"] = True
        return not st["stopped"]

    for rid in [r for r in rows if r not in elig]:                    # no longer eligible
        _remove_reel(col, collection, rid)
        st["removed"] += 1

    todo = []
    for rid, e in elig.items():
        row = rows.get(rid)
        if row and row["canonical_attempt_id"] == e["aid"] and row["status"] in ("indexed", "empty"):
            st["skipped"] += 1
        else:
            todo.append(rid)

    batch: list[tuple[str, dict, list[dict], str]] = []

    def flush() -> None:
        if not batch:
            return
        if not go_on():                                # idle check failed: no new embedding batch
            st["deferred"] += len(batch)
            batch.clear()
            return
        texts = [chunking.embed_text(c) for _, _, chunks, _ in batch for c in chunks]
        try:
            from . import models
            with models.heavy("index embedding"):        # never alongside answer generation or graph extraction
                vecs = emb.encode_docs(texts) if texts else []
        except Exception as ex:  # noqa: BLE001 — record, never crash the sync
            for rid, e, _, h in batch:
                _write_error(collection, rid, e["aid"], rows.get(rid), f"embedding failed: {ex}")
                st["errors"] += 1
            batch.clear()
            return
        k = 0
        for rid, e, chunks, h in batch:
            _apply(col, collection, rid, e["aid"], chunks, vecs[k:k + len(chunks)], h)
            k += len(chunks)
            st["embedded"] += 1
            st["chunks_written"] += len(chunks)
        batch.clear()

    for rid in todo:
        if not go_on():
            break
        e = elig[rid]
        try:
            chunks = chunking.build_chunks(load_record(e["aid"]), e["source"])
            h = chunking.content_hash(chunks)
        except Exception as ex:  # noqa: BLE001
            _write_error(collection, rid, e["aid"], rows.get(rid), f"chunking failed: {ex}")
            st["errors"] += 1
            continue
        row = rows.get(rid)
        if row and row["content_hash"] == h and row["status"] in ("indexed", "empty"):
            _write_state(collection, rid, e["aid"], h, json.loads(row["chunk_ids"] or "[]"), row["status"])
            st["unchanged_hash"] += 1
            continue
        batch.append((rid, e, chunks, h))
        if sum(len(b[2]) for b in batch) >= EMBED_BATCH_CHUNKS:
            flush()
    if st["stopped"]:
        st["deferred"] += len(batch)
        batch.clear()
    else:
        flush()
    if not st["stopped"]:                               # cheap at this scale (one metadata scan)
        st["orphans_deleted"] = sweep_orphans(col, collection, elig)
    st["duration_s"] = round(time.time() - t0, 2)
    return st


def _changed(st: dict) -> bool:
    return any(st[k] for k in ("embedded", "unchanged_hash", "removed", "orphans_deleted", "errors"))


# ---------------------------------------------------------------- verification

def verify(collection: str) -> dict:
    """Is `collection` complete and consistent with the eligible canonical records in SQLite right now?"""
    col = get_collection(collection)
    spec = spec_of(col)
    emb = embedder.for_spec(spec)
    elig = eligible()
    rows = state_rows(collection)
    problems: list[str] = []
    expected_ids: dict[str, str] = {}               # chunk id → expected content hash
    missing_rows = sorted(set(elig) - set(rows))
    extra_rows = sorted(set(rows) - set(elig))
    error_rows = sorted(r for r, v in rows.items() if v["status"] == "error")
    stale_rows, hash_rows = [], []
    for rid, e in elig.items():
        row = rows.get(rid)
        if not row or row["status"] == "error":
            continue
        if row["canonical_attempt_id"] != e["aid"]:
            stale_rows.append(rid)
            continue
        chunks = chunking.build_chunks(load_record(e["aid"]), e["source"])
        h = chunking.content_hash(chunks)
        if row["content_hash"] != h or json.loads(row["chunk_ids"] or "[]") != [c["id"] for c in chunks]:
            hash_rows.append(rid)
        for c in chunks:
            expected_ids[c["id"]] = h
    got = col.get(include=["metadatas"])
    ids = got["ids"]
    dup = len(ids) - len(set(ids))
    meta = dict(zip(ids, got["metadatas"]))
    missing = sorted(set(expected_ids) - set(ids))
    orphans = sorted(set(ids) - set(expected_ids))
    bad_hash = sorted(i for i in set(ids) & set(expected_ids) if (meta[i] or {}).get("content_hash") != expected_ids[i])
    for label, items in (("eligible Reels without index rows", missing_rows), ("rows for ineligible Reels", extra_rows),
                         ("Reels with indexing errors", error_rows), ("rows behind the canonical record", stale_rows),
                         ("rows whose content changed", hash_rows), ("chunks missing from Chroma", missing),
                         ("orphan chunks in Chroma", orphans), ("chunks with a stale content hash", bad_hash)):
        if items:
            problems.append(f"{len(items)} {label}: {', '.join(items[:5])}{' …' if len(items) > 5 else ''}")
    if dup:
        problems.append(f"{dup} duplicate chunk ids")
    if ids:
        sample = col.get(ids=[ids[0]], include=["embeddings"])["embeddings"]
        if len(sample[0]) != emb.dim:
            problems.append(f"embedding dimension {len(sample[0])} ≠ model dimension {emb.dim}")
        probe = col.query(query_embeddings=[emb.encode_query("test").tolist()], n_results=1)
        if not probe["ids"] or not probe["ids"][0]:
            problems.append("probe query returned no results")
    return {"collection": collection, "ok": not problems, "problems": problems,
            "counts": {"eligible": len(elig), "rows": len(rows), "chunks": len(ids), "expected_chunks": len(expected_ids)}}


# ---------------------------------------------------------------- rebuild / sync orchestration

def _set_build(state: str, **kw) -> None:
    jobs.set_meta(K_BUILD, json.dumps({"state": state, "at": _now(), **kw}))


def build(force: bool = False, should_continue=None) -> dict:
    """Build (or resume building) a collection for the current spec; activate it only once verified."""
    with writing():
        _ensure_schema()
        spec = embedder.current_spec()
        sig = embedder.signature(spec)
        active = active_collection()
        building = jobs.get_meta(K_BUILDING)
        if building and (building not in collections()
                         or embedder.signature(spec_of(get_collection(building))) != sig):
            if building in collections():
                _drop_collection(building)                     # a build for an outdated spec (never searched)
            jobs.set_meta(K_BUILDING, "")
            building = None
        if not force and not building and active and embedder.signature(spec_of(get_collection(active))) == sig:
            return {"state": "current", "collection": active}
        if not building:
            building = _create_collection(spec)
            jobs.set_meta(K_BUILDING, building)
        _set_build("building", collection=building, replaces=active)

        result = {"collection": building, "replaces": active, "passes": []}
        for _ in range(VERIFY_ROUNDS):
            for _ in range(CATCHUP_MAX_PASSES):                # catch up with SQLite until a pass changes nothing
                st = sync_collection(building, should_continue)
                result["passes"].append(st)
                if st["stopped"]:
                    _set_build("paused", collection=building, replaces=active)
                    return {**result, "state": "paused"}
                if not _changed(st):
                    break
            v = verify(building)
            result["verification"] = v
            if v["ok"]:
                break
        else:
            _set_build("failed_verification", collection=building, replaces=active, problems=v["problems"])
            return {**result, "state": "failed_verification"}

        with _readers_lock, jobs.tx() as con:                  # atomic switch
            con.execute("INSERT INTO meta(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (K_ACTIVE, building))
            con.execute("DELETE FROM meta WHERE key=?", (K_BUILDING,))
        if active and active != building:
            _retire(active)                                     # dropped later by cleanup_retired()
        _set_build("activated", collection=building, replaced=active)
        return {**result, "state": "activated"}


def sync_all(rebuild: bool = False, should_continue=None) -> dict:
    """Keep the active collection current; build/resume a new one when needed (or forced)."""
    with writing():
        _ensure_schema()
        t0 = time.time()
        out: dict = {"at": _now(), "retired_dropped": cleanup_retired()}
        known = set(collections())
        retired = {r["name"] for r in _meta_json(K_RETIRED) or []}
        with jobs.tx() as con:                                  # bookkeeping for collections that no longer exist
            for (name,) in con.execute("SELECT DISTINCT collection FROM index_state").fetchall():
                if name not in known and name not in retired:
                    con.execute("DELETE FROM index_state WHERE collection=?", (name,))
        active = active_collection()
        if active:
            out["active"] = sync_collection(active, should_continue)
            if out["active"]["stopped"]:
                out["duration_s"] = round(time.time() - t0, 2)
                jobs.set_meta(K_LAST, json.dumps(out))
                return out
        spec = embedder.current_spec()
        needs = rebuild or not active or jobs.get_meta(K_BUILDING) or \
            embedder.signature(spec_of(get_collection(active))) != embedder.signature(spec)
        if needs:
            out["build"] = build(force=rebuild, should_continue=should_continue)
        out["duration_s"] = round(time.time() - t0, 2)
        jobs.set_meta(K_LAST, json.dumps(out))
        return out


# ---------------------------------------------------------------- idle gate + background thread

def busy_reason(quiet_s: float | None = None, now: float | None = None) -> str | None:
    """Why automatic indexing must not run now (None = idle)."""
    now = now or time.time()
    spool = worker.spool_dir()
    if spool.exists():
        live = [p for p in spool.iterdir() if p.name != ".dead"]      # queued, processing or being uploaded
        if live:
            return f"processing ({len(live)} job(s) in the spool)"
    with _db() as con:
        open_sessions = con.execute(
            "SELECT COUNT(*) FROM sessions WHERE finished_at IS NULL AND COALESCE(last_seen, created_at, 0) > ?",
            (now - jobs.STALE_SESSION_S,)).fetchone()[0]
        if open_sessions:
            return "capture session in progress"
        if quiet_s:
            s = con.execute("SELECT MAX(COALESCE(last_seen, 0)), MAX(COALESCE(finished_at, 0)), "
                            "MAX(COALESCE(created_at, 0)) FROM sessions").fetchone()
            last = max([x or 0 for x in s] or [0])
            a = con.execute("SELECT MAX(processed_at) FROM attempts").fetchone()[0]
            if a:
                try:
                    last = max(last, datetime.fromisoformat(a).timestamp())
                except ValueError:
                    pass
            if now - last < quiet_s:
                return f"waiting for a quiet period ({int(quiet_s - (now - last))} s left)"
    return None


_thread: threading.Thread | None = None
_stop = threading.Event()
_wake = threading.Event()
_requests: list[dict] = []
_state = {"running": False, "busy": None, "last_error": None}


def _run(fn, **kw) -> dict | None:
    _state["running"] = True
    try:
        out = fn(**kw)
        _state["last_error"] = None
        return out
    except Exception as ex:  # noqa: BLE001
        _state["last_error"] = f"{type(ex).__name__}: {ex}"
        print(f"[indexer] {_state['last_error']}", file=sys.stderr)
        return None
    finally:
        _state["running"] = False


def _loop() -> None:
    while not _stop.is_set():
        _wake.wait(POLL_S)
        _wake.clear()
        if _stop.is_set():
            break
        if not acquire_writer():
            _state["busy"] = "another process holds the index writer lock"
            continue
        if _requests:
            req = _requests.pop(0)
            _state["busy"] = None
            _run(sync_all, rebuild=req.get("rebuild", False), should_continue=lambda: not _stop.is_set())
            continue
        _run(cleanup_retired)
        if not config.INDEX_AUTO:
            _state["busy"] = "automatic indexing disabled (INDEX_AUTO=0)"
            continue
        reason = busy_reason(config.INDEX_IDLE_S)
        _state["busy"] = reason
        if reason is None:
            _run(sync_all, should_continue=lambda: not _stop.is_set() and busy_reason() is None)


def start() -> None:
    global _thread
    _stop.clear()
    acquire_writer()                    # best effort; the loop retries if another process holds it
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_loop, name="reel-indexer", daemon=True)
        _thread.start()
    _wake.set()


def stop() -> bool:
    """Stop the indexer thread. The writer lock is released only once the thread has exited (and no write
    is in progress); otherwise it stays held so no second writer can start. → fully stopped?"""
    _stop.set()
    _wake.set()
    if _thread is not None:
        _thread.join(timeout=STOP_JOIN_S)
        if _thread.is_alive():
            print("[indexer] still finishing a write; keeping the index writer lock", file=sys.stderr)
            return False
    return release_writer()


def request_sync(rebuild: bool = False, wait: bool = False) -> dict:
    """Manual sync: always allowed (even while capture runs)."""
    if wait or _thread is None or not _thread.is_alive():
        return sync_all(rebuild=rebuild)
    _requests.append({"rebuild": rebuild})
    _wake.set()
    return {"accepted": True, "queued_requests": len(_requests)}


# ---------------------------------------------------------------- search

def _where(source: str | None, types: list[str] | None) -> dict | None:
    clauses = []
    if source and source != "all":
        clauses.append({"source": source})
    if types:
        kinds = sorted({k for t in types for k in TYPE_GROUPS.get(t, [t])})
        clauses.append({"content_type": {"$in": kinds}})
    return None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _hit(cid: str, doc: str, md: dict, dist: float) -> dict:
    sim = 1.0 - float(dist)
    weight = config.FRAGMENT_WEIGHT if "fragment" in (md.get("ocr_quality") or "") else 1.0
    return {"chunk_id": cid, "reel_id": md.get("reel_id"), "content_type": md.get("content_type"), "text": doc,
            "start": md.get("start"), "end": md.get("end"), "score": round(sim * weight, 4),
            "similarity": round(sim, 4), "ocr_quality": md.get("ocr_quality"), "ocr_echo": md.get("ocr_echo"),
            "ocr_kinds": md.get("ocr_kinds"), "has_low_conf": md.get("has_low_conf"),
            "passage_ids": json.loads(md.get("passage_ids") or "[]"),
            "_reel": {k: md.get(k) for k in ("source", "source_url", "owner_username", "caption_short", "saved_at",
                                             "record_status")}}


def search(q: str, k: int = 10, source: str | None = "instagram", types: list[str] | None = None,
           group: str = "reel", per_reel: int = 3) -> dict:
    with reading() as name:
        if not name:
            b = _meta_json(K_BUILD)
            raise IndexNotReady("no index built yet" + (f" (build {b['state']})" if b else "; run a sync"))
        col = get_collection(name)
        spec = spec_of(col)
        emb = embedder.for_spec(spec)
        total = col.count()
        hits: list[dict] = []
        if total and q.strip():
            n = min(total, max(50, k * 4 * (per_reel if group == "reel" else 1)))
            res = col.query(query_embeddings=[emb.encode_query(q).tolist()], n_results=n,
                            where=_where(source, types), include=["documents", "metadatas", "distances"])
            hits = sorted((_hit(*x) for x in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                                res["distances"][0])), key=lambda h: -h["score"])
    out = {"query": q, "collection": name, "model": spec["model"], "group": group}
    if group == "none":
        out["results"] = [{**{k_: v for k_, v in h.items() if k_ != "_reel"}, **h["_reel"]} for h in hits[:k]]
        return out
    reels: dict[str, dict] = {}
    for h in hits:
        r = reels.setdefault(h["reel_id"], {"reel_id": h["reel_id"], **h["_reel"], "score": h["score"], "passages": []})
        if len(r["passages"]) < per_reel:
            r["passages"].append({k_: v for k_, v in h.items() if k_ not in ("_reel", "reel_id")})
    out["results"] = list(reels.values())[:k]
    return out


# ---------------------------------------------------------------- status

def status() -> dict:
    _ensure_schema()
    active = active_collection()
    out = {"active": active, "building": jobs.get_meta(K_BUILDING) or None, "build": _meta_json(K_BUILD),
           "retired": [r["name"] for r in _meta_json(K_RETIRED) or []], "last_sync": _meta_json(K_LAST),
           "auto": config.INDEX_AUTO, "running": _state["running"], "busy": _state["busy"],
           "last_error": _state["last_error"], "configured_model": config.EMBED_MODEL}
    elig = eligible()
    if active:
        col = get_collection(active)
        spec = spec_of(col)
        rows = state_rows(active)
        ok = {r for r, v in rows.items() if v["status"] in ("indexed", "empty")
              and r in elig and v["canonical_attempt_id"] == elig[r]["aid"]}
        out.update(model=spec["model"], revision=spec["revision"], signature=embedder.signature(spec),
                   chunks=col.count(), eligible=len(elig), indexed=len(ok),
                   stale=sum(1 for r, v in rows.items() if r in elig and v["canonical_attempt_id"] != elig[r]["aid"]),
                   pending=sum(1 for r in elig if r not in rows),
                   errors=sum(1 for v in rows.values() if v["status"] == "error"))
    else:
        out.update(eligible=len(elig), indexed=0, pending=len(elig))
    return out


def _main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "status":
        print(json.dumps(status(), indent=1))
    elif cmd in ("sync", "rebuild"):
        if not acquire_writer():
            print("The backend is running and owns the index. Use:  curl -X POST "
                  f"'http://localhost:{config.PORT}/api/index/sync{'?rebuild=1' if cmd == 'rebuild' else ''}'")
            return 2
        try:
            print(json.dumps(sync_all(rebuild=cmd == "rebuild"), indent=1))
        finally:
            release_writer()
    elif cmd == "verify":
        targets = [c for c in (active_collection(), jobs.get_meta(K_BUILDING)) if c and c in collections()]
        if not targets:
            print("no collection to verify")
            return 1
        results = [verify(c) for c in targets]
        print(json.dumps(results, indent=1))
        return 0 if all(r["ok"] for r in results) else 1
    elif cmd == "search":
        q = argv[2] if len(argv) > 2 else ""
        k = int(argv[3]) if len(argv) > 3 else 5
        for i, r in enumerate(search(q, k=k)["results"], 1):
            print(f"{i}. [{r['score']:.3f}] {r['source_url']}  @{r['owner_username']}  {r['caption_short'][:70]}")
            for p in r["passages"]:
                t = f"{p['start']:.0f}–{p['end']:.0f}s" if p.get("start") is not None else ""
                print(f"     {p['content_type']:<12} {t:<9} {p['score']:.3f}  {' '.join(p['text'].split())[:110]}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
