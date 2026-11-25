"""Incremental, recoverable graph sync from SQLite canonical records to Neo4j."""
from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import re
import sys
import threading
import time
from contextlib import contextmanager

from .. import chunking, config, db, embedder, indexer, jobs, models
from . import extract as ex
from . import normalize as nz
from .store import GraphOffline, Neo4jStore

MAX_EXTRACT_ATTEMPTS = 3
MAX_BUSY_RETRIES = 3
POLL_S = 120
STOP_JOIN_S = 60
CHUNK_ID_RE = re.compile(r"^(?P<reel>.+):(?P<kind>[a-z_]+):(?P<seq>\d{3,})$")
REVIEW_FIELDS = ("subject", "predicate", "object", "chunk_id", "decision", "note")
SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_extraction (
  reel_id TEXT NOT NULL, content_hash TEXT NOT NULL, extractor_version TEXT NOT NULL, canonical_attempt_id INTEGER,
  status TEXT NOT NULL, raw TEXT, error TEXT, stats TEXT, attempts INTEGER NOT NULL DEFAULT 1, created_at TEXT,
  PRIMARY KEY (reel_id, content_hash, extractor_version));
CREATE TABLE IF NOT EXISTS graph_state (
  reel_id TEXT PRIMARY KEY, canonical_attempt_id INTEGER, content_hash TEXT, extractor_version TEXT,
  normalizer_version TEXT, graph_hash TEXT, chunk_ids TEXT, status TEXT NOT NULL, error TEXT, synced_at TEXT);
CREATE TABLE IF NOT EXISTS graph_review (
  subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, chunk_id TEXT NOT NULL,
  decision TEXT NOT NULL, note TEXT, reviewed_at TEXT, PRIMARY KEY (subject, predicate, object, chunk_id));
"""
K_LAST = "graph.last_sync"

class SyncBusy(RuntimeError):
    pass

def ensure_schema() -> None:
    with jobs.connect() as con:
        con.executescript(SCHEMA)

def extractor_version(model_id: str | None = None) -> str:
    model_id = model_id or config.GRAPH_LLM
    return ex.extractor_version(model_id, embedder.resolve_revision(model_id))

def accepted_versions(llm=None) -> tuple[str, list[str]]:
    ev = getattr(llm, "version", None) or extractor_version()
    if hasattr(llm, "compatible_versions"):
        compat = list(llm.compatible_versions)
    elif getattr(llm, "version", None):
        compat = []
    else:
        compat = compatible_versions()
    return ev, [ev, *[v for v in compat if v != ev]]

def compatible_versions(model_id: str | None = None) -> list[str]:
    model_id = model_id or config.GRAPH_LLM
    try:
        return ex.compatible_versions(model_id, embedder.resolve_revision(model_id))
    except Exception:  # noqa: BLE001
        return []

def eligible() -> dict[str, dict]:
    return {r: e for r, e in indexer.eligible().items() if e["source"] in config.GRAPH_SOURCES}

_sync_lock = threading.Lock()

@contextmanager
def exclusive():
    if not _sync_lock.acquire(blocking=False):
        raise SyncBusy("a graph sync is already running in this process")
    fd = None
    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fd = open(config.OUTPUT_DIR / ".graph-sync.lock", "a+")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SyncBusy("a graph sync is already running in another process")
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                fd.close()
        _sync_lock.release()
        if _requests:
            _wake.set()

def states() -> dict[str, dict]:
    ensure_schema()
    with jobs.connect() as con:
        return {r["reel_id"]: dict(r) for r in con.execute("SELECT * FROM graph_state")}

_STATE_COLS = ["canonical_attempt_id", "content_hash", "extractor_version", "normalizer_version", "graph_hash",
               "chunk_ids", "status", "error"]

def _upsert_state(con, rid: str, **f) -> None:
    con.execute(f"""INSERT INTO graph_state(reel_id, {', '.join(_STATE_COLS)}, synced_at)
                    VALUES(?, {', '.join('?' * len(_STATE_COLS))}, ?)
                    ON CONFLICT(reel_id) DO UPDATE SET {', '.join(f'{c}=excluded.{c}' for c in _STATE_COLS)},
                    synced_at=excluded.synced_at""",
                (rid, *[f.get(c) for c in _STATE_COLS], db.utc_now()))

def _put_state(rid: str, **f) -> None:
    with jobs.tx() as con:
        _upsert_state(con, rid, **f)

def _review_rows(con, chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    return [dict(r) for r in con.execute(
        f"""SELECT {', '.join(REVIEW_FIELDS)} FROM graph_review WHERE chunk_id IN ({','.join('?' * len(chunk_ids))})
            ORDER BY subject, predicate, object, chunk_id""", chunk_ids)]

def _digest_rows(rows: list[dict]) -> str:
    key = sorted((tuple(r[f] for f in REVIEW_FIELDS) for r in rows), key=lambda t: t[:4])
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()

def _commit_state_if_reviews_unchanged(rid: str, review_chunk_ids: list[str], digest: str, **f) -> bool:
    with jobs.tx() as con:
        if _digest_rows(_review_rows(con, review_chunk_ids)) == digest:
            _upsert_state(con, rid, **f)
            return True
        _upsert_state(con, rid, **{**f, "graph_hash": None, "status": "stale",
                                   "error": "a review changed during the sync; re-applied next sync"})
        return False

def _set_status(rid: str, expected: str, status: str) -> bool:
    with jobs.tx() as con:
        return con.execute("""UPDATE graph_state SET status=?, synced_at=? WHERE reel_id=? AND status=?
                              AND graph_hash IS NOT NULL""", (status, db.utc_now(), rid, expected)).rowcount == 1

def _mark_error(rid: str, prev: dict | None, msg: str) -> None:
    if prev:
        with jobs.tx() as con:
            con.execute("UPDATE graph_state SET status='error', error=?, synced_at=? WHERE reel_id=?", (msg, db.utc_now(), rid))
    else:
        _put_state(rid, status="error", error=msg)

def _delete_state(rid: str) -> None:
    with jobs.tx() as con:
        con.execute("DELETE FROM graph_state WHERE reel_id=?", (rid,))

def cached(rid: str, content_hash: str, ev: str) -> dict | None:
    ensure_schema()
    with jobs.connect() as con:
        r = con.execute("SELECT * FROM graph_extraction WHERE reel_id=? AND content_hash=? AND extractor_version=?",
                        (rid, content_hash, ev)).fetchone()
    return dict(r) if r else None

def usable_cache(rid: str, content_hash: str, versions: list[str]) -> dict | None:
    for v in versions:
        c = cached(rid, content_hash, v)
        if c is not None and c["status"] == "ok":
            return c
    return None

def _put_cache(rid: str, content_hash: str, ev: str, aid: int, status: str, raw=None, error=None, stats=None) -> None:
    with jobs.tx() as con:
        con.execute("""INSERT INTO graph_extraction(reel_id, content_hash, extractor_version, canonical_attempt_id,
                         status, raw, error, stats, attempts, created_at) VALUES(?,?,?,?,?,?,?,?,1,?)
                       ON CONFLICT(reel_id, content_hash, extractor_version) DO UPDATE SET status=excluded.status,
                         raw=excluded.raw, error=excluded.error, stats=excluded.stats,
                         canonical_attempt_id=excluded.canonical_attempt_id, created_at=excluded.created_at,
                         attempts=graph_extraction.attempts + 1""",
                    (rid, content_hash, ev, aid, status, json.dumps(raw) if raw is not None else None, error,
                     json.dumps(stats or {}), db.utc_now()))

def reviews_for(chunk_ids: list[str]) -> tuple[dict[tuple, dict], str]:
    ensure_schema()
    with jobs.connect() as con:
        rows = _review_rows(con, chunk_ids)
    return {(r["subject"], r["predicate"], r["object"], r["chunk_id"]): r for r in rows}, _digest_rows(rows)

def owner_of(chunk_id: str) -> str | None:
    """Parsed from the right of `{reel_id}:{kind}:{seq}` when unrecorded, so Reel ids are never cut short."""
    ensure_schema()
    with jobs.connect() as con:
        for r in con.execute("SELECT reel_id, chunk_ids FROM graph_state WHERE chunk_ids LIKE ?",
                             (f'%{json.dumps(chunk_id)}%',)):
            if chunk_id in json.loads(r["chunk_ids"] or "[]"):
                return r["reel_id"]
    m = CHUNK_ID_RE.match(chunk_id)
    return m.group("reel") if m else None

def put_review(subject: str, predicate: str, obj: str, chunk_id: str, decision: str, note: str | None) -> str:
    if decision not in ("accept", "reject"):
        raise ValueError("decision must be accept or reject")
    rid = owner_of(chunk_id)
    if rid is None:
        raise ValueError(f"cannot determine the Reel that owns chunk {chunk_id!r}")
    ensure_schema()
    with jobs.tx() as con:
        con.execute("""INSERT INTO graph_review(subject, predicate, object, chunk_id, decision, note, reviewed_at)
                       VALUES(?,?,?,?,?,?,?) ON CONFLICT(subject, predicate, object, chunk_id) DO UPDATE SET
                       decision=excluded.decision, note=excluded.note, reviewed_at=excluded.reviewed_at""",
                    (subject, predicate, obj, chunk_id, decision, note, db.utc_now()))
        con.execute("UPDATE graph_state SET graph_hash=NULL, status='stale' WHERE reel_id=?", (rid,))
    return rid

def build_doc(record: dict, source: str, prepared: dict, grounded: dict, ev: str, reviews: dict,
              partial: bool = False) -> dict:
    chunks = chunking.build_chunks(record, source)
    evidence = {p["chunk_id"] for p in prepared["passages"]}
    md = record.get("metadata") or {}
    owner, caption_short = chunking._context(record)
    reel = {"reel_id": record["reel_id"], "source": source, "source_url": record.get("source_url") or "",
            "owner": owner, "caption_short": caption_short, "saved_at": record.get("saved_at") or "",
            "canonical_attempt_id": record.get("_attempt_id"), "content_hash": prepared["content_hash"],
            "extractor_version": ev, "normalizer_version": grounded["normalizer_version"],
            "chunker_version": chunking.CHUNKER_VERSION, "hashtags": md.get("hashtags") or []}
    if partial:
        reel["extraction_partial"] = True
    passages = []
    for c in chunks:
        m = c["metadata"]
        p = {"chunk_id": c["id"], "reel_id": record["reel_id"], "content_type": m["content_type"], "text": c["text"],
             "evidence": c["id"] in evidence, "passage_ids": m["passage_ids"], "ocr_quality": m.get("ocr_quality", "")}
        if m.get("start") is not None:
            p["start"], p["end"] = m["start"], m.get("end")
        passages.append(p)
    concepts = [{"key": c["key"], "name": c["name"], "type": c["type"], "curated": nz.curated_type(c["name"]) is not None}
                for c in sorted(grounded["concepts"].values(), key=lambda x: x["key"])]
    mentions = [{**m, "extractor_version": ev} for m in grounded["mentions"]]
    assertions = []
    for a in grounded["assertions"]:
        rv = reviews.get((a["subject"], a["predicate"], a["object"], a["chunk_id"]))
        status = {"accept": "verified", "reject": "rejected"}[rv["decision"]] if rv else "proposed"
        assertions.append({**a, "status": status, "review_note": (rv or {}).get("note"), "extractor_version": ev})
    doc = {"reel": reel, "passages": passages, "concepts": concepts, "mentions": mentions, "assertions": assertions}
    doc["graph_hash"] = hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return doc

def chroma_ready(chunk_ids: list[str], content_hash: str) -> bool:
    try:
        name = indexer.active_collection()
        if not name or not chunk_ids:
            return False
        got = indexer.get_collection(name).get(ids=list(chunk_ids), include=["metadatas"])
        return len(set(got["ids"])) == len(set(chunk_ids)) and \
            all((m or {}).get("content_hash") == content_hash for m in got["metadatas"])
    except Exception:  # noqa: BLE001
        return False

def sync(store=None, llm=None, allow_extract: bool = True, force: bool = False, should_continue=None,
         only: list[str] | None = None) -> dict:
    with exclusive():
        return _sync(store, llm, allow_extract, force, should_continue, only)

def _sync(store, llm, allow_extract, force, should_continue, only) -> dict:
    t0 = time.time()
    ensure_schema()
    ev, accepted = accepted_versions(llm)
    nv = ex.grounding_version()
    elig = eligible()
    if only:
        elig = {r: e for r, e in elig.items() if r in only}
    st = {"eligible": len(elig), "skipped": 0, "extracted": 0, "extract_errors": 0, "from_cache": 0, "written": 0,
          "write_errors": 0, "ready": 0, "awaiting_index": 0, "promoted": 0, "demoted": 0, "deferred": 0,
          "removed": 0, "stale": 0, "not_retried": 0, "stopped": False, "online": True, "extractor_version": ev,
          "accepted_extractor_versions": accepted, "normalizer_version": nv}
    tm: dict[str, float] = {k: 0.0 for k in ("prepare_s", "model_load_s", "llm_s", "ground_s", "chroma_s",
                                             "neo4j_s", "sqlite_s")}

    @contextmanager
    def timed(k):
        t = time.perf_counter()
        try:
            yield
        finally:
            tm[k] += time.perf_counter() - t

    store = store or Neo4jStore()

    def offline(err) -> None:
        st["online"], st["offline_reason"] = False, str(err)

    try:
        with timed("neo4j_s"):
            store.ensure_schema()
            hashes = store.reel_hashes()
    except GraphOffline as e:
        offline(e)
        hashes = {}
    with timed("sqlite_s"):
        rows = states()
    own_llm = llm is None

    if st["online"] and not only:
        for rid in sorted((set(hashes) | set(rows)) - set(elig)):
            try:
                if rid in hashes:
                    store.remove_reel(rid)
                _delete_state(rid)
                st["removed"] += 1
            except GraphOffline as e:
                offline(e)
                break

    try:
        for rid, e in elig.items():
            if should_continue is not None and not should_continue():
                st["stopped"] = True
                break
            s = rows.get(rid)
            unchanged = (not force and s and s["status"] in ("ready", "awaiting_index")
                         and s["canonical_attempt_id"] == e["aid"] and s["extractor_version"] in accepted
                         and s["normalizer_version"] == nv)
            if unchanged and (not st["online"] or hashes.get(rid) == s["graph_hash"]):
                st["skipped"] += 1
                with timed("chroma_s"):
                    ready_now = chroma_ready(json.loads(s["chunk_ids"] or "[]"), s["content_hash"])
                want = "ready" if ready_now else "awaiting_index"
                if want != s["status"]:
                    if not st["online"]:
                        st[s["status"]] += 1
                        continue
                    try:
                        store.set_ready(rid, ready_now)
                    except GraphOffline as err:
                        offline(err)
                        st[s["status"]] += 1
                        continue
                    if not _set_status(rid, s["status"], want):
                        st["stale"] += 1
                        continue
                    st["promoted" if ready_now else "demoted"] += 1
                st[want] += 1
                continue

            with timed("prepare_s"):
                record = indexer.load_record(e["aid"])
                record["_attempt_id"] = e["aid"]
                prepared = ex.prepare(record, e["source"])
            with timed("sqlite_s"):
                c = usable_cache(rid, prepared["content_hash"], accepted)
            if c is None:
                with timed("sqlite_s"):
                    failed = cached(rid, prepared["content_hash"], ev)
                if failed is not None and not force:
                    deterministic = (failed["error"] or "").startswith("ExtractionError")
                    if deterministic or failed["attempts"] >= MAX_EXTRACT_ATTEMPTS:
                        _mark_error(rid, s, f"extraction failed {failed['attempts']}× ({failed['error']}); "
                                            "not retried automatically — retry with force")
                        st["extract_errors"] += 1
                        st["not_retried"] += 1
                        continue
                if not allow_extract:
                    st["deferred"] += 1
                    continue
                if llm is None:
                    llm = models.shared_llm(config.GRAPH_LLM)
                try:
                    # Never concurrent with Chroma embedding (writer lock) or other model work in any process (heavy lock).
                    with indexer._write_lock, models.heavy("graph extraction"), timed("llm_s"):
                        raw, stats = ex.propose(llm, prepared)
                except Exception as err:  # noqa: BLE001
                    msg = f"{type(err).__name__}: {err}"
                    _put_cache(rid, prepared["content_hash"], ev, e["aid"], "error", error=msg,
                               stats={"attempts": getattr(err, "attempts", []), "recovery": ex.RECOVERY_VERSION})
                    _mark_error(rid, s, f"extraction: {msg}")
                    st["extract_errors"] += 1
                    continue
                _put_cache(rid, prepared["content_hash"], ev, e["aid"], "ok", raw=raw, stats=stats)
                used_ev, used_stats = ev, stats
                st["extracted"] += 1
            else:
                raw = json.loads(c["raw"])
                used_ev, used_stats = c["extractor_version"], json.loads(c["stats"] or "{}")
                st["from_cache"] += 1
            if not st["online"]:
                continue
            with timed("ground_s"):
                grounded = ex.ground(raw, prepared)
                all_ids = [x["id"] for x in chunking.build_chunks(record, e["source"])]
            with timed("sqlite_s"):
                reviews, digest = reviews_for(all_ids)
            doc = build_doc(record, e["source"], prepared, grounded, used_ev, reviews,
                            partial=bool(used_stats.get("partial")))
            with timed("chroma_s"):
                ready = chroma_ready(all_ids, prepared["content_hash"])
            doc["retrieval_ready"] = ready
            try:
                with timed("neo4j_s"):
                    if hashes.get(rid) != doc["graph_hash"]:
                        store.write_reel(doc)
                        st["written"] += 1
                    else:
                        store.set_ready(rid, ready)
            except GraphOffline as err:
                offline(err)
                continue
            except Exception as err:  # noqa: BLE001
                _mark_error(rid, s, f"graph write: {type(err).__name__}: {err}")
                st["write_errors"] += 1
                continue
            hashes[rid] = doc["graph_hash"]
            with timed("sqlite_s"):
                committed = _commit_state_if_reviews_unchanged(
                    rid, all_ids, digest, canonical_attempt_id=e["aid"], content_hash=prepared["content_hash"],
                    extractor_version=used_ev, normalizer_version=nv, graph_hash=doc["graph_hash"],
                    chunk_ids=json.dumps(all_ids), status="ready" if ready else "awaiting_index", error=None)
            if committed:
                st["ready" if ready else "awaiting_index"] += 1
            else:
                st["stale"] += 1
    finally:
        if llm is not None:
            tm["model_load_s"] = float(getattr(llm, "load_seconds", 0) or 0)
        if own_llm and llm is not None:
            llm.unload()
    st["timing"] = {k: round(v, 2) for k, v in tm.items()}
    st["duration_s"] = round(time.time() - t0, 2)
    jobs.set_meta(K_LAST, json.dumps({"at": db.utc_now(), **st}))
    return st

_thread: threading.Thread | None = None
_stop = threading.Event()
_wake = threading.Event()
_requests: list[dict] = []
_request_log: dict[int, dict] = {}
_ids = itertools.count(1)
_state = {"running": False, "busy": None, "last_error": None}

def _run(**kw) -> dict | None:
    _state["running"] = True
    try:
        out = sync(**kw)
        _state["last_error"] = None
        return out
    except SyncBusy:
        raise
    except Exception as e:  # noqa: BLE001
        _state["last_error"] = f"{type(e).__name__}: {e}"
        print(f"[graph] {_state['last_error']}", file=sys.stderr)
        return None
    finally:
        _state["running"] = False

def _idle() -> bool:
    return not _stop.is_set() and indexer.busy_reason() is None and not models.interactive_waiting()

def _process_request(req: dict) -> None:
    rec = _request_log[req["id"]]
    rec.update(status="running", started_at=db.utc_now())
    busy = indexer.busy_reason()
    try:
        out = _run(allow_extract=req["force"] or busy is None, force=req["force"],
                   should_continue=lambda: not _stop.is_set())
    except SyncBusy as e:
        req["busy_retries"] += 1
        if req["busy_retries"] >= MAX_BUSY_RETRIES:
            rec.update(status="failed", reason=f"{e} ({req['busy_retries']} attempts)", at=db.utc_now())
        else:
            rec.update(status="queued", reason=str(e))
            _requests.insert(0, req)
        return
    rec.update(status="done" if out is not None else "failed", result=out,
               reason=None if out is not None else _state["last_error"], at=db.utc_now())

def _loop() -> None:
    while not _stop.is_set():
        _wake.wait(POLL_S)
        _wake.clear()
        if _stop.is_set():
            break
        if _requests:
            _process_request(_requests.pop(0))
            continue
        if not config.GRAPH_AUTO:
            _state["busy"] = "automatic graph sync disabled (GRAPH_AUTO=0)"
            continue
        reason = indexer.busy_reason(config.INDEX_IDLE_S) or \
            ("semantic indexing running" if indexer._state["running"] else None)
        _state["busy"] = reason
        if reason is None:
            try:
                _run(should_continue=_idle)
            except SyncBusy as e:
                _state["busy"] = str(e)

def start() -> None:
    global _thread
    _stop.clear()
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_loop, name="reel-graph", daemon=True)
        _thread.start()

def stop() -> bool:
    _stop.set()
    _wake.set()
    if _thread is not None:
        _thread.join(timeout=STOP_JOIN_S)
        return not _thread.is_alive()
    return True

def request_status(request_id: int) -> dict | None:
    return _request_log.get(request_id)

def request_sync(force: bool = False, wait: bool = False) -> dict:
    if wait or _thread is None or not _thread.is_alive():
        try:
            return sync(allow_extract=force or indexer.busy_reason() is None, force=force)
        except SyncBusy as e:
            return {"busy": True, "reason": str(e)}
    rid = next(_ids)
    _request_log[rid] = {"id": rid, "status": "queued", "force": force, "queued_at": db.utc_now()}
    _requests.append({"id": rid, "force": force, "busy_retries": 0})
    _wake.set()
    return {"accepted": True, "request_id": rid, "queued_requests": len(_requests)}