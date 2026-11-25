"""Graph inspection, verification and Phase-4B hooks (read-only Cypher over Neo4jStore).

Every result carries reel_id, source_url, chunk_id, start/end and (for relations) the verbatim quote, so it can
be traced back to the saved Reel. Relation status is always shown:
  proposed      model-extracted, quote verified against the passage (provenance, not truth)
  corroborated  the same subject-predicate-object proposed independently in ≥ 2 Reels (still not verified)
  verified / rejected   human review (rejected ones are hidden unless include_rejected)
Co-occurrence ("mentioned in the same Reel") is reported separately and never as a relationship.
"""
from __future__ import annotations

import json

from .. import indexer
from . import normalize as nz
from . import sync as gs
from .store import Neo4jStore

STATUS_EXPR = """CASE WHEN a.status IN ['verified','rejected'] THEN a.status
                      WHEN support >= 2 THEN 'corroborated' ELSE 'proposed' END"""


def resolve(store: Neo4jStore, name: str) -> dict | None:
    """Concept by canonical key, else by name/alias (case-insensitive), else best full-text hit."""
    key = nz.canonicalize(name)[0]
    rows = store.run("MATCH (c:Concept {key: $k}) RETURN c.key AS key, c.name AS name, c.type AS type, "
                     "c.aliases AS aliases", k=key)
    if rows:
        return rows[0]
    rows = store.run("""MATCH (c:Concept) WHERE toLower(c.name) = toLower($n) OR
                        any(a IN coalesce(c.aliases, []) WHERE toLower(a) = toLower($n))
                        RETURN c.key AS key, c.name AS name, c.type AS type, c.aliases AS aliases LIMIT 1""", n=name)
    if rows:
        return rows[0]
    try:
        rows = store.run("""CALL db.index.fulltext.queryNodes('concept_names', $q) YIELD node, score
                            RETURN node.key AS key, node.name AS name, node.type AS type, node.aliases AS aliases
                            ORDER BY score DESC LIMIT 1""", q=name.replace("/", " "))
    except Exception:  # noqa: BLE001 — odd query syntax
        rows = []
    return rows[0] if rows else None


def counts(store: Neo4jStore) -> dict:
    nodes = {r["label"]: r["n"] for r in store.run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n")}
    rels = {r["t"]: r["n"] for r in store.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n")}
    status = {r["s"]: r["n"] for r in store.run(f"""
        MATCH (a:Assertion) OPTIONAL MATCH (b:Assertion {{subject_key: a.subject_key, predicate: a.predicate,
                                                        object_key: a.object_key}})
        WITH a, count(DISTINCT b.reel_id) AS support RETURN {STATUS_EXPR} AS s, count(*) AS n""")}
    ready = {str(r["ready"]): r["n"] for r in store.run(
        "MATCH (r:Reel) RETURN coalesce(r.retrieval_ready, false) AS ready, count(*) AS n")}
    return {"nodes": nodes, "relationships": rels, "assertion_status": status, "reels_ready": ready}


def status(store: Neo4jStore | None = None) -> dict:
    store = store or Neo4jStore()
    rows = gs.states()
    by = {}
    for s in rows.values():
        by[s["status"]] = by.get(s["status"], 0) + 1
    out = {"eligible": len(gs.eligible()), "sqlite_state": by, "background": dict(gs._state),
           "last_sync": json.loads(gs.jobs.get_meta(gs.K_LAST) or "null")}
    try:
        out["neo4j"] = {"online": True, **counts(store)}
    except gs.GraphOffline as e:
        out["neo4j"] = {"online": False, "reason": str(e)}
    return out


def reel_concepts(store: Neo4jStore, reel_id: str) -> dict:
    reel = store.run("MATCH (r:Reel {reel_id: $rid}) RETURN r {.*} AS r", rid=reel_id)
    concepts = store.run("""
        MATCH (r:Reel {reel_id: $rid})-[:HAS_PASSAGE]->(p:Passage)-[m:MENTIONS]->(c:Concept)
        WITH c, collect({chunk_id: p.chunk_id, content_type: p.content_type, start: p.start, end: p.end,
                         surface: m.surface, method: m.method}) AS ev
        OPTIONAL MATCH (c)<-[:MENTIONS]-(:Passage)<-[:HAS_PASSAGE]-(o:Reel) WHERE o.reel_id <> $rid
        RETURN c.key AS key, c.name AS name, c.type AS type, size(ev) AS passages, ev[0..3] AS evidence,
               count(DISTINCT o) AS other_reels
        ORDER BY passages DESC, name""", rid=reel_id)
    assertions = evidence(store, reel_id=reel_id)
    return {"reel": reel[0]["r"] if reel else None, "concepts": concepts, "assertions": assertions}


def concept_reels(store: Neo4jStore, name: str) -> dict:
    c = resolve(store, name)
    if not c:
        return {"concept": None, "reels": []}
    reels = store.run("""
        MATCH (c:Concept {key: $k})<-[m:MENTIONS]-(p:Passage)<-[:HAS_PASSAGE]-(r:Reel)
        WITH r, collect({chunk_id: p.chunk_id, content_type: p.content_type, start: p.start, end: p.end,
                         surface: m.surface, text: left(p.text, 240)}) AS ev
        RETURN r.reel_id AS reel_id, r.source_url AS source_url, r.owner AS owner, r.caption_short AS caption,
               r.retrieval_ready AS retrieval_ready, size(ev) AS passages, ev[0..2] AS evidence
        ORDER BY passages DESC""", k=c["key"])
    return {"concept": c, "reels": reels}


def shared(store: Neo4jStore, min_reels: int = 2, limit: int = 50) -> list[dict]:
    return store.run("""
        MATCH (c:Concept)<-[:MENTIONS]-(:Passage)<-[:HAS_PASSAGE]-(r:Reel)
        WITH c, collect(DISTINCT {reel_id: r.reel_id, url: r.source_url}) AS reels
        WHERE size(reels) >= $m
        RETURN c.key AS key, c.name AS name, c.type AS type, size(reels) AS n_reels, reels[0..8] AS reels
        ORDER BY n_reels DESC, name LIMIT $lim""", m=min_reels, lim=limit)


def evidence(store: Neo4jStore, a: str | None = None, b: str | None = None, reel_id: str | None = None,
             include_rejected: bool = False) -> list[dict]:
    """Assertions (both directions between a and b, or all of one Reel) with their evidence passage."""
    params: dict = {"inc": include_rejected}
    if reel_id:
        where = "a.reel_id = $rid"
        params["rid"] = reel_id
    else:
        ca, cb = resolve(store, a), resolve(store, b)
        if not ca or not cb:
            return []
        where = "((a.subject_key = $ka AND a.object_key = $kb) OR (a.subject_key = $kb AND a.object_key = $ka))"
        params.update(ka=ca["key"], kb=cb["key"])
    return store.run(f"""
        MATCH (a:Assertion)-[:EVIDENCE]->(p:Passage)<-[:HAS_PASSAGE]-(r:Reel) WHERE {where}
        MATCH (a)-[:SUBJECT]->(s:Concept), (a)-[:OBJECT]->(o:Concept)
        OPTIONAL MATCH (b:Assertion {{subject_key: a.subject_key, predicate: a.predicate, object_key: a.object_key}})
        WITH a, p, r, s, o, count(DISTINCT b.reel_id) AS support
        WITH a, p, r, s, o, support, {STATUS_EXPR} AS status
        WHERE $inc OR status <> 'rejected'
        RETURN a.id AS assertion_id, s.name AS subject, a.predicate AS predicate, o.name AS object, status,
               support AS supporting_reels, a.method AS method, a.confidence AS confidence, a.quote AS quote,
               a.review_note AS review_note, r.reel_id AS reel_id, r.source_url AS source_url, p.chunk_id AS chunk_id,
               p.content_type AS content_type, p.start AS start, p.end AS end, left(p.text, 300) AS passage
        ORDER BY CASE status WHEN 'verified' THEN 0 WHEN 'corroborated' THEN 1 WHEN 'proposed' THEN 2 ELSE 3 END,
                 subject, predicate""", **params)


def between(store: Neo4jStore, a: str, b: str, include_rejected: bool = False) -> dict:
    ca, cb = resolve(store, a), resolve(store, b)
    if not ca or not cb:
        return {"a": ca, "b": cb, "relations": {}, "co_mentioned_in": [], "two_hop": []}
    rel = evidence(store, a, b, include_rejected=include_rejected)
    grouped: dict[str, list] = {}
    for x in rel:
        grouped.setdefault(x["status"], []).append(x)
    co = store.run("""
        MATCH (ra:Reel)-[:HAS_PASSAGE]->(:Passage)-[:MENTIONS]->(:Concept {key: $ka})
        MATCH (ra)-[:HAS_PASSAGE]->(:Passage)-[:MENTIONS]->(:Concept {key: $kb})
        RETURN DISTINCT ra.reel_id AS reel_id, ra.source_url AS source_url""", ka=ca["key"], kb=cb["key"])
    hops = store.run("""
        MATCH (x:Concept {key: $ka})<-[:SUBJECT|OBJECT]-(a1:Assertion)-[:SUBJECT|OBJECT]->(m:Concept)
              <-[:SUBJECT|OBJECT]-(a2:Assertion)-[:SUBJECT|OBJECT]->(y:Concept {key: $kb})
        WHERE m.key <> $ka AND m.key <> $kb AND a1.status <> 'rejected' AND a2.status <> 'rejected'
        RETURN DISTINCT m.name AS via, a1.predicate AS first, a1.reel_id AS first_reel,
               a2.predicate AS second, a2.reel_id AS second_reel LIMIT 20""", ka=ca["key"], kb=cb["key"])
    return {"a": ca, "b": cb, "relations": grouped,
            "co_mentioned_in": co, "co_mention_note": "same Reel only; not a relationship",
            "two_hop": hops, "two_hop_note": "paths through model-proposed assertions; not verified"}


# ---------------------------------------------------------------- Phase 4B hooks (helpers only)

def concepts_for_chunks(store: Neo4jStore, chunk_ids: list[str], ready_only: bool = True) -> list[dict]:
    """Chroma hits → their concepts (only Reels whose graph matches the active index, by default)."""
    return store.run("""
        MATCH (r:Reel)-[:HAS_PASSAGE]->(p:Passage)-[m:MENTIONS]->(c:Concept)
        WHERE p.chunk_id IN $ids AND (NOT $ready OR r.retrieval_ready)
        RETURN p.chunk_id AS chunk_id, c.key AS key, c.name AS name, c.type AS type, m.surface AS surface""",
                     ids=list(chunk_ids), ready=ready_only)


def neighbors(store: Neo4jStore, keys: list[str], include_proposed: bool = True, limit: int = 50) -> list[dict]:
    """One hop over assertions from the given concepts (statuses returned; rejected excluded)."""
    return store.run(f"""
        MATCH (c:Concept)<-[r1:SUBJECT|OBJECT]-(a:Assertion)-[r2:SUBJECT|OBJECT]->(n:Concept)
        WHERE c.key IN $keys AND n.key <> c.key AND a.status <> 'rejected'
        OPTIONAL MATCH (b:Assertion {{subject_key: a.subject_key, predicate: a.predicate, object_key: a.object_key}})
        WITH c, n, a, count(DISTINCT b.reel_id) AS support
        WITH c, n, a, support, {STATUS_EXPR} AS status
        WHERE $prop OR status <> 'proposed'
        RETURN c.key AS from, a.predicate AS predicate, n.key AS to, n.name AS name, status, support,
               a.reel_id AS reel_id, a.chunk_id AS chunk_id LIMIT $lim""", keys=list(keys), prop=include_proposed, lim=limit)


# ---------------------------------------------------------------- verification

def verify(store: Neo4jStore | None = None, check_chroma: bool = True) -> dict:
    """SQLite eligible set vs graph_state vs Neo4j; orphans; provenance; duplicates; Chroma cross-check."""
    from .extract import grounding_version as ex_grounding
    from .extract import norm_text, surface_match

    store = store or Neo4jStore()
    problems: list[str] = []
    elig = gs.eligible()
    rows = gs.states()
    graph = {r["id"]: r for r in store.run(
        "MATCH (r:Reel) RETURN r.reel_id AS id, r.graph_hash AS h, r.canonical_attempt_id AS aid, "
        "r.retrieval_ready AS ready")}

    def report(label, items):
        items = sorted(items)
        if items:
            problems.append(f"{len(items)} {label}: {', '.join(map(str, items[:5]))}{' …' if len(items) > 5 else ''}")

    report("eligible Reels missing from the graph", set(elig) - set(graph))
    report("Reels in the graph that are not eligible", set(graph) - set(elig))
    report("Reels without graph_state", set(graph) - set(rows))
    report("graph_state rows whose graph hash differs from Neo4j",
           [r for r, s in rows.items() if r in graph and s["graph_hash"] and s["graph_hash"] != graph[r]["h"]])
    report("Reels behind their canonical record", [r for r, g in graph.items() if r in elig and g["aid"] != elig[r]["aid"]])
    report("Reels in error state", [r for r, s in rows.items() if s["status"] == "error"])
    _, accepted = gs.accepted_versions()
    report("Reels extracted by an extractor version that is no longer accepted",
           [r for r, s in rows.items() if s["status"] != "error" and s["extractor_version"] not in accepted])
    report("Reels grounded with an outdated normalization/grounding version",
           [r for r, s in rows.items() if s["status"] != "error" and s["normalizer_version"] != ex_grounding()])
    report("Reels marked stale (review pending re-apply)", [r for r, s in rows.items() if s["status"] == "stale"])
    report("graph_state readiness differs from Neo4j",
           [r for r, s in rows.items() if r in graph and s["status"] in ("ready", "awaiting_index")
            and bool(graph[r]["ready"]) != (s["status"] == "ready")])
    orphans = store.run("""
        CALL { MATCH (p:Passage) WHERE NOT (p)<-[:HAS_PASSAGE]-(:Reel) RETURN 'passage without Reel' AS k, p.chunk_id AS id
               UNION MATCH (c:Concept) WHERE NOT (c)<-[:MENTIONS]-() AND NOT (c)<-[:SUBJECT|OBJECT]-()
                     RETURN 'concept without mentions or assertions' AS k, c.key AS id
               UNION MATCH (a:Assertion) WHERE NOT (a)-[:SUBJECT]->() OR NOT (a)-[:OBJECT]->() OR NOT (a)-[:EVIDENCE]->()
                     RETURN 'assertion missing subject/object/evidence' AS k, a.id AS id
               UNION MATCH (a:Assertion)-[:EVIDENCE]->(p:Passage) WHERE p.reel_id <> a.reel_id
                     RETURN 'assertion evidence from another Reel' AS k, a.id AS id }
        RETURN k, collect(id) AS ids""")
    for o in orphans:
        report(o["k"], o["ids"])
    bad_mentions = [f"{m['chunk_id']}→{m['key']}" for m in store.run(
        "MATCH (p:Passage)-[m:MENTIONS]->(c:Concept) RETURN p.chunk_id AS chunk_id, p.text AS text, "
        "m.surface AS surface, c.key AS key") if not surface_match(m["surface"], m["text"])]
    report("mentions whose surface is not in the passage", bad_mentions)
    bad_quotes = [a["id"] for a in store.run(
        "MATCH (a:Assertion)-[:EVIDENCE]->(p:Passage) RETURN a.id AS id, a.quote AS q, p.text AS text")
        if f" {norm_text(a['q'])} " not in f" {norm_text(a['text'])} "]
    report("assertions whose quote is not in the evidence passage", bad_quotes)
    dup = store.run("""MATCH (c:Concept) WITH toLower(replace(replace(c.name, '-', ''), ' ', '')) AS n, collect(c.key) AS keys
                       WHERE size(keys) > 1 RETURN n, keys""")
    from .normalize import contextual_pairs
    distinct = contextual_pairs()                     # e.g. physics "higgs field" vs the company "higgsfield"
    near = [f"{d['keys']}" for d in dup if not _distinct_identifiers(d["keys"]) and frozenset(d["keys"]) not in distinct]
    report("near-duplicate concepts (case/spacing collisions)", near)
    if check_chroma:
        name = indexer.active_collection()
        if name:
            ids = {p["id"] for p in store.run("MATCH (p:Passage) RETURN p.chunk_id AS id")}
            have = set(indexer.get_collection(name).get(ids=sorted(ids), include=[])["ids"]) if ids else set()
            missing = ids - have
            ready_missing = {i for i in missing if graph.get(gs.owner_of(i) or "", {}).get("ready")}
            report("passages of 'ready' Reels missing from the active Chroma collection", ready_missing)
            if missing - ready_missing:
                problems.append(f"(info) {len(missing - ready_missing)} passages of awaiting_index Reels not yet in Chroma")
    partial = [r["id"] for r in store.run("MATCH (r:Reel) WHERE r.extraction_partial RETURN r.reel_id AS id")]
    if partial:
        problems.append(f"(info) {len(partial)} Reels with partial extraction (model output truncated; only fully "
                        f"emitted concepts/relations kept): {', '.join(sorted(partial))}")
    versions = {}
    for s in rows.values():
        versions[s["extractor_version"]] = versions.get(s["extractor_version"], 0) + 1
    hard = [p for p in problems if not p.startswith("(info)")]
    return {"ok": not hard, "problems": problems, "reels_in_graph": len(graph), "eligible": len(elig),
            "extractor_versions": versions}


def _distinct_identifiers(keys: list[str]) -> bool:
    """Keys that only collide after stripping hyphens/spaces but are deliberately distinct (cuda-q vs cudaq is a
    real duplicate; c++ vs c is not a collision at all)."""
    return len({k.replace("-", "").replace(" ", "") for k in keys}) == len(keys)
