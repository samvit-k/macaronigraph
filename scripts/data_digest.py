"""Read-only production data digest (Phase 4C integrity check). Fails closed: any source that cannot be verified
makes the digest INCOMPLETE (exit 2), and an incomplete digest is never reported as IDENTICAL.

  .venv/bin/python scripts/data_digest.py --out reports/phase4c/integrity_before.json [--with-verify]
  .venv/bin/python scripts/data_digest.py --compare A.json B.json

Exit codes: 0 complete (or IDENTICAL) · 1 DIFFERENT · 2 INCOMPLETE (a source or verification failed) · 3 BUSY
(production ingestion/indexing/heavy work active — nothing is interrupted; retry when idle).

Covers:
  - pipeline.db canonical tables (reels, attempts), SQLite mode=ro;
  - Chroma: every table of chroma.sqlite3 (incl. the embeddings_queue vectors, metadata, FTS), mode=ro, AND every
    file of every HNSW segment directory (data_level0.bin, header.bin, length.bin, link_lists.bin,
    index_metadata.pickle), hashed byte-for-byte. No Chroma client is created by the digest itself;
  - Neo4j: counts per label/type, a hash of every node (labels + properties) and every relationship identified by
    type + both endpoints' canonical ids (Label:key — Reel.reel_id, Passage.chunk_id, Concept.key, Assertion.id) +
    properties, through a READ-access session. A node without exactly one canonical id, or two nodes with the same
    id, makes the digest incomplete. Credentials come from backend.config (environment / .env.local), never printed.
  - Idle check before and after the snapshot (read-only): live spool entries, open capture sessions, and
    non-blocking probes of the heavy-work and Chroma writer locks (an existing lock file is opened read-only; a
    missing one is not created).
  - --with-verify additionally records `indexer verify` and `graph verify` (separate processes; they use the normal
    clients, so run them with the backend stopped) and re-digests afterwards to prove verification changed nothing.
Derived state that is meant to change (conversations.db, the graph_extraction cache, logs) is not digested.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.jobs import STALE_SESSION_S  # noqa: E402  (same staleness rule as the backend)

CANONICAL_TABLES = ("reels", "attempts")
# Chroma writes one (id, lock_status) row into acquire_write every time a client opens the store (including
# `indexer verify`). It carries no Reel data; it is still hashed and reported, but excluded from the verdict — and
# only while its schema is exactly that lock log, so data can never hide there.
CHROMA_BOOKKEEPING = {"acquire_write": ["id", "lock_status"]}
NODE_KEYS = {"Reel": "reel_id", "Passage": "chunk_id", "Concept": "key", "Assertion": "id"}

class Incomplete(RuntimeError):
    pass

# ---------------------------------------------------------------- helpers

@contextlib.contextmanager
def _ro(path: Path):
    if not path.is_file():
        raise Incomplete(f"missing database: {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        yield con
    finally:
        con.close()

def _table_digest(con: sqlite3.Connection, table: str) -> dict:
    h, n = hashlib.sha256(), 0
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
    order = ", ".join(f'"{c}"' for c in cols) or "rowid"
    for row in con.execute(f'SELECT * FROM "{table}" ORDER BY {order}'):
        h.update(repr(tuple(row)).encode())
        n += 1
    return {"rows": n, "columns": cols, "sha256": h.hexdigest()}

def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# ---------------------------------------------------------------- sources

def canonical(output_dir: Path) -> dict:
    with _ro(output_dir / "pipeline.db") as con:
        have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = [t for t in CANONICAL_TABLES if t not in have]
        if missing:
            raise Incomplete(f"canonical tables missing: {missing}")
        return {t: _table_digest(con, t) for t in CANONICAL_TABLES}

def chroma(chroma_dir: Path) -> dict:
    with _ro(chroma_dir / "chroma.sqlite3") as con:
        tables = sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        out = {"sqlite": {t: _table_digest(con, t) for t in tables}, "segments": {}}
    for t, cols in CHROMA_BOOKKEEPING.items():
        if t in out["sqlite"] and out["sqlite"][t]["columns"] != cols:
            raise Incomplete(f"Chroma table {t} no longer has the expected lock-log schema {cols}")
    segs = sorted(p for p in chroma_dir.iterdir() if p.is_dir())
    if not segs:
        raise Incomplete(f"no HNSW segment directories in {chroma_dir}")
    for seg in segs:
        files = sorted(f for f in seg.iterdir() if f.is_file())
        if not files:
            raise Incomplete(f"empty segment directory {seg.name}")
        out["segments"][seg.name] = {f.name: {"bytes": f.stat().st_size, "sha256": _file_sha(f)} for f in files}
    return out

def _canonical_id(labels: list[str], props: dict) -> str:
    ids = [f"{lb}:{props[NODE_KEYS[lb]]}" for lb in sorted(labels) if lb in NODE_KEYS and props.get(NODE_KEYS[lb]) is not None]
    if len(ids) != 1:
        raise Incomplete(f"node with labels {sorted(labels)} has {len(ids)} canonical ids")
    return ids[0]

NODES_Q = "MATCH (n) RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props"
RELS_Q = ("MATCH (a)-[r]->(b) RETURN type(r) AS type, elementId(a) AS a, elementId(b) AS b, "
          "properties(r) AS props")

def neo4j_from_rows(nodes: list[dict], rels: list[dict]) -> dict:
    """Pure function over query rows (testable without a server)."""
    cid_of, seen, node_lines, labels = {}, set(), [], {}
    for n in nodes:
        cid = _canonical_id(n["labels"], n["props"])
        if cid in seen:
            raise Incomplete(f"duplicate canonical id {cid}")
        seen.add(cid)
        cid_of[n["eid"]] = cid
        for lb in n["labels"]:
            labels[lb] = labels.get(lb, 0) + 1
        node_lines.append(json.dumps([cid, sorted(n["labels"]), n["props"]], sort_keys=True, default=str))
    rel_lines, types = [], {}
    for r in rels:
        if r["a"] not in cid_of or r["b"] not in cid_of:
            raise Incomplete(f"relationship {r['type']} with an unidentified endpoint")
        types[r["type"]] = types.get(r["type"], 0) + 1
        rel_lines.append(json.dumps([r["type"], cid_of[r["a"]], cid_of[r["b"]], r["props"]], sort_keys=True, default=str))
    return {"labels": dict(sorted(labels.items())), "relationships": dict(sorted(types.items())),
            "nodes_sha256": hashlib.sha256("\n".join(sorted(node_lines)).encode()).hexdigest(),
            "relationships_sha256": hashlib.sha256("\n".join(sorted(rel_lines)).encode()).hexdigest()}

def neo4j(query=None) -> dict:
    """query(cypher) → list of dict rows; default: a READ-access session on the configured server."""
    if query is None:
        try:
            from neo4j import READ_ACCESS, GraphDatabase
        except ImportError as e:
            raise Incomplete(f"neo4j driver missing: {e}") from e
        if not config.NEO4J_PASSWORD:
            raise Incomplete("NEO4J_PASSWORD is not set")
        drv = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        try:
            with drv.session(default_access_mode=READ_ACCESS) as s:
                rows = {q: [dict(r) for r in s.run(q)] for q in (NODES_Q, RELS_Q)}
        except Exception as e:  # noqa: BLE001
            raise Incomplete(f"neo4j unavailable: {type(e).__name__}") from e
        finally:
            drv.close()
        return neo4j_from_rows(rows[NODES_Q], rows[RELS_Q])
    return neo4j_from_rows(query(NODES_Q), query(RELS_Q))

# ---------------------------------------------------------------- idle check (read-only)

def _lock_held(path: Path) -> bool:
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)

def busy(output_dir: Path, now: float | None = None) -> list[str]:
    now = now or time.time()
    out = []
    spool = output_dir / "spool"
    if spool.is_dir():
        live = [p for p in spool.iterdir() if p.name != ".dead"]
        if live:
            out.append(f"{len(live)} job(s) in the spool")
    db = output_dir / "pipeline.db"
    if db.is_file():
        with _ro(db) as con:
            have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "sessions" in have:
                n = con.execute("SELECT COUNT(*) FROM sessions WHERE finished_at IS NULL AND "
                                "COALESCE(last_seen, created_at, 0) > ?", (now - STALE_SESSION_S,)).fetchone()[0]
                if n:
                    out.append(f"{n} open capture session(s)")
    for name, p in (("heavy work (model/embedding)", output_dir / ".heavy-work.lock"),
                    ("Chroma writer", output_dir / "chroma" / ".writer.lock")):
        if _lock_held(p):
            out.append(f"{name} lock is held")
    return out

# ---------------------------------------------------------------- digest / verify / compare

def digest(output_dir: Path | None = None, chroma_dir: Path | None = None, query=None) -> dict:
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    chroma_dir = Path(chroma_dir or output_dir / "chroma")
    out = {"taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "errors": {}}
    for name, fn in (("canonical", lambda: canonical(output_dir)), ("chroma", lambda: chroma(chroma_dir)),
                     ("neo4j", lambda: neo4j(query))):
        try:
            out[name] = fn()
        except (Incomplete, sqlite3.Error, OSError) as e:
            out[name] = None
            out["errors"][name] = str(e)
    out["complete"] = not out["errors"]
    return out

def run_verify() -> dict:
    res = {}
    for name, mod in (("indexer_verify", "backend.indexer"), ("graph_verify", "backend.graph")):
        p = subprocess.run([sys.executable, "-m", mod, "verify"], cwd=ROOT, capture_output=True, text=True, timeout=600)
        try:
            data = json.loads(p.stdout)
        except json.JSONDecodeError:
            data = None
        items = data if isinstance(data, list) else [data]
        ok = p.returncode == 0 and data is not None and all(isinstance(x, dict) and x.get("ok") for x in items)
        res[name] = {"ok": ok, "returncode": p.returncode, "result": data}
    return res

def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v

def compare(a: dict, b: dict) -> tuple[str, list[str]]:
    problems = [f"{n}: incomplete ({', '.join(x.get('errors') or {}) or 'not marked complete'})"
                for n, x in (("first", a), ("second", b)) if not x.get("complete") or x.get("errors")]
    for n, x in (("first", a), ("second", b)):
        for v, r in (x.get("verify") or {}).items():
            if not r.get("ok"):
                problems.append(f"{n}: {v} not ok")
    if problems:
        return "INCOMPLETE", problems
    keys = ("canonical", "chroma", "neo4j")
    fa, fb = dict(_flat({k: a[k] for k in keys})), dict(_flat({k: b[k] for k in keys}))
    changed = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    book = tuple(f"chroma.sqlite.{t}." for t in CHROMA_BOOKKEEPING)
    diff = [f"{k}: {fa.get(k)} → {fb.get(k)}" for k in changed if not k.startswith(book)]
    notes = [f"(bookkeeping, ignored) {k}: {fa.get(k)} → {fb.get(k)}" for k in changed
             if k.startswith(book) and not k.endswith(".sha256")]
    return ("DIFFERENT", diff + notes) if diff else ("IDENTICAL", notes)

EXIT = {"IDENTICAL": 0, "DIFFERENT": 1, "INCOMPLETE": 2}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--with-verify", action="store_true")
    a = ap.parse_args(argv)
    if a.compare:
        verdict, lines = compare(*(json.loads(Path(p).read_text()) for p in a.compare))
        print("\n".join([verdict] + lines))
        return EXIT[verdict]
    out_dir = Path(config.OUTPUT_DIR)
    before = busy(out_dir)
    if before:
        print("BUSY: " + "; ".join(before) + " — nothing was interrupted; retry when idle")
        return 3
    d = digest(out_dir)
    if a.with_verify and d["complete"]:
        d["verify"] = run_verify()
        again = digest(out_dir)
        verdict, lines = compare(d, again)
        d["unchanged_by_verify"] = verdict == "IDENTICAL"
        if verdict != "IDENTICAL":
            d["errors"]["verify"] = f"digest after verification: {verdict} {lines[:3]}"
        for v, r in d["verify"].items():
            if not r["ok"]:
                d["errors"][v] = "not ok"
    after = busy(out_dir)
    if after:
        d["errors"]["idle"] = "became busy during the snapshot: " + "; ".join(after)
    d["complete"] = not d["errors"]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(d, indent=1) + "\n")
    summary = {"complete": d["complete"], "errors": d["errors"],
               "canonical": {t: v["rows"] for t, v in (d["canonical"] or {}).items()},
               "chroma_segments": list((d["chroma"] or {}).get("segments", {})),
               "neo4j": (d["neo4j"] or {}).get("labels"),
               "verify": {k: v["ok"] for k, v in (d.get("verify") or {}).items()}}
    print(json.dumps(summary, indent=1))
    return 0 if d["complete"] else 2

if __name__ == "__main__":
    sys.exit(main())