"""Graph CLI:  python -m backend.graph <command>

  status                         counts, readiness, last sync, Neo4j availability
  sync [--force]                 incremental sync (LLM extraction only when capture/processing are idle, or --force)
  verify                         SQLite ⇄ graph_state ⇄ Neo4j ⇄ Chroma consistency, orphans, provenance, duplicates
  reel <reel_id>                 concepts and assertions of one Reel
  concept <name>                 Reels mentioning a concept (with evidence passages)
  shared [min_reels]             concepts appearing in ≥ min_reels Reels (default 2)
  between <a> <b>                relations between two concepts, grouped by status; co-mentions shown separately
  evidence <a> <b> [--include-rejected]   every supporting passage + quote for relations between a and b
  review <assertion_id> accept|reject [note]   human decision (kept in SQLite, survives refreshes/rebuilds)
  candidates [threshold]         possible duplicate concepts by name-embedding similarity (suggestions only)
"""
from __future__ import annotations

import json
import sys

from . import queries as q
from . import sync as gs
from .store import GraphOffline, Neo4jStore


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def candidates(store: Neo4jStore, threshold: float = 0.9) -> list[dict]:
    """Concept-name pairs whose Qwen3 embeddings are very similar. Suggestions for aliases.json only — the graph
    is never changed by this (similarity is not evidence)."""
    import numpy as np

    from .. import embedder
    rows = store.run("MATCH (c:Concept) RETURN c.key AS key, c.name AS name")
    if len(rows) < 2:
        return []
    emb = embedder.for_spec(embedder.current_spec())
    vecs = emb.encode_docs([r["name"] for r in rows])
    sims = vecs @ vecs.T
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if sims[i, j] >= threshold:
                out.append({"a": rows[i]["name"], "b": rows[j]["name"], "similarity": round(float(sims[i, j]), 3)})
    return sorted(out, key=lambda x: -x["similarity"])


def main(argv: list[str]) -> int:
    cmd, args = (argv[1] if len(argv) > 1 else "status"), argv[2:]
    store = Neo4jStore()
    try:
        if cmd == "status":
            _print(q.status(store))
        elif cmd == "sync":
            force = "--force" in args
            from .. import indexer
            busy = indexer.busy_reason()
            if busy and not force:
                print(f"note: {busy} — cached extractions are written, new LLM extraction is deferred (use --force)",
                      file=sys.stderr)
            _print(gs.sync(store=store, allow_extract=force or busy is None, force=force))
        elif cmd == "verify":
            v = q.verify(store)
            _print(v)
            return 0 if v["ok"] else 1
        elif cmd == "reel" and args:
            _print(q.reel_concepts(store, args[0]))
        elif cmd == "concept" and args:
            _print(q.concept_reels(store, " ".join(args)))
        elif cmd == "shared":
            _print(q.shared(store, int(args[0]) if args else 2))
        elif cmd == "between" and len(args) >= 2:
            _print(q.between(store, args[0], args[1], include_rejected="--include-rejected" in args))
        elif cmd == "evidence" and len(args) >= 2:
            _print(q.evidence(store, args[0], args[1], include_rejected="--include-rejected" in args))
        elif cmd == "review" and len(args) >= 2 and args[1] in ("accept", "reject"):
            a = store.run("MATCH (a:Assertion {id: $id}) RETURN a.subject_key AS s, a.predicate AS p, "
                          "a.object_key AS o, a.chunk_id AS c", id=args[0])
            if not a:
                print(f"no assertion {args[0]!r}", file=sys.stderr)
                return 1
            a = a[0]
            rid = gs.put_review(a["s"], a["p"], a["o"], a["c"], args[1], " ".join(args[2:]) or None)
            try:
                _print({"review": args[1], "reel_id": rid, "applied": gs.sync(store=store, only=[rid], allow_extract=False)})
            except gs.SyncBusy as e:
                _print({"review": args[1], "reel_id": rid, "applied": False, "note": f"{e}; applied on the next sync"})
        elif cmd == "candidates":
            _print(candidates(store, float(args[0]) if args else 0.9))
        else:
            print(__doc__)
            return 2
    except GraphOffline as e:
        print(f"graph offline: {e}", file=sys.stderr)
        return 3
    except gs.SyncBusy as e:
        print(f"busy: {e}", file=sys.stderr)
        return 4
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
