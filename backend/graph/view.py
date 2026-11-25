"""Read-only graph views for the MacaroniGraph workspace (Phase 5). No writes, no new graph generation.

The workspace never loads the whole graph: it starts from an OVERVIEW (Reels, the "bridge" concepts that several Reels
mention, and the extracted assertions) and expands one NEIGHBORHOOD at a time. That keeps the canvas usable when the
collection grows to thousands of Reels.

Vocabulary, kept strictly apart (as in queries.py):
  mention    a Reel's passage mentions a concept (Reel -> Concept). Co-mention is NOT a relationship.
  assertion  a model-extracted subject-predicate-object with a verbatim quote; status proposed / corroborated /
             verified (rejected assertions are never returned here).
"""
from __future__ import annotations

from .queries import STATUS_EXPR, resolve
from .store import Neo4jStore

EXCERPT = 240

_ASSERTIONS = f"""
    MATCH (a:Assertion)-[:SUBJECT]->(s:Concept), (a)-[:OBJECT]->(o:Concept)
    WHERE {{where}}
    OPTIONAL MATCH (b:Assertion {{{{subject_key: a.subject_key, predicate: a.predicate, object_key: a.object_key}}}})
    WITH a, s, o, count(DISTINCT b.reel_id) AS support
    WITH a, s, o, support, {STATUS_EXPR} AS status
    WHERE status <> 'rejected'
    RETURN a.id AS id, s.key AS subject_key, s.name AS subject, a.predicate AS predicate, o.key AS object_key,
           o.name AS object, status, support AS supporting_reels, a.reel_id AS reel_id, a.chunk_id AS chunk_id,
           a.quote AS quote, a.confidence AS confidence
    ORDER BY id"""

def _assertions(store: Neo4jStore, where: str = "true", **params) -> list[dict]:
    return store.run(_ASSERTIONS.format(where=where), **params)

def _concept_rows(store: Neo4jStore, keys: list[str]) -> list[dict]:
    """Concepts by key with their GLOBAL Reel count (how many Reels mention them)."""
    if not keys:
        return []
    return store.run("""
        MATCH (c:Concept) WHERE c.key IN $keys
        OPTIONAL MATCH (c)<-[:MENTIONS]-(:Passage)<-[:HAS_PASSAGE]-(r:Reel)
        RETURN c.key AS key, c.name AS name, c.type AS type, coalesce(c.aliases, []) AS aliases,
               count(DISTINCT r) AS n_reels""", keys=keys)

def _reel_fields() -> str:
    return ("r.reel_id AS reel_id, r.owner AS owner, r.caption_short AS caption, r.source_url AS source_url, "
            "r.saved_at AS saved_at")

def overview(store: Neo4jStore, min_reels: int = 2, limit_concepts: int = 150, reel_offset: int = 0,
             reel_limit: int = 200) -> dict:
    """Reels (paged, newest first), bridge concepts mentioned by >= min_reels of them, their mention edges, and the
    non-rejected assertions whose concepts are shown."""
    totals = {
        "reels": store.run("MATCH (r:Reel) RETURN count(r) AS n")[0]["n"],
        "concepts": store.run("MATCH (c:Concept) RETURN count(c) AS n")[0]["n"],
        "bridge": store.run("""MATCH (c:Concept)<-[:MENTIONS]-(:Passage)<-[:HAS_PASSAGE]-(r:Reel)
                               WITH c, count(DISTINCT r) AS n WHERE n >= $m RETURN count(c) AS n""", m=min_reels)[0]["n"],
    }
    reels = store.run(f"""
        MATCH (r:Reel)
        OPTIONAL MATCH (r)-[:HAS_PASSAGE]->(:Passage)-[:MENTIONS]->(c:Concept)
        WITH r, count(DISTINCT c) AS concepts
        RETURN {_reel_fields()}, concepts
        ORDER BY r.saved_at DESC, r.reel_id SKIP $off LIMIT $lim""", off=reel_offset, lim=reel_limit)
    rids = [r["reel_id"] for r in reels]
    bridge = store.run("""
        MATCH (c:Concept)<-[:MENTIONS]-(:Passage)<-[:HAS_PASSAGE]-(r:Reel)
        WITH c, count(DISTINCT r) AS n_reels WHERE n_reels >= $m
        RETURN c.key AS key, c.name AS name, c.type AS type, coalesce(c.aliases, []) AS aliases, n_reels
        ORDER BY n_reels DESC, name LIMIT $lim""", m=min_reels, lim=limit_concepts)
    assertions = _assertions(store)
    keys = {c["key"] for c in bridge}
    extra = sorted({k for a in assertions for k in (a["subject_key"], a["object_key"])} - keys)
    concepts = [{**c, "bridge": True} for c in bridge] + [{**c, "bridge": c["n_reels"] >= min_reels}
                                                          for c in _concept_rows(store, extra)]
    shown = [c["key"] for c in concepts]
    mentions = store.run("""
        MATCH (r:Reel)-[:HAS_PASSAGE]->(p:Passage)-[:MENTIONS]->(c:Concept)
        WHERE r.reel_id IN $rids AND c.key IN $keys
        RETURN r.reel_id AS reel_id, c.key AS concept_key, count(p) AS passages""", rids=rids, keys=shown)
    return {"reels": reels, "concepts": concepts, "mentions": mentions, "assertions": assertions,
            "totals": {**totals, "reels_shown": len(reels), "concepts_shown": len(concepts),
                       "reel_offset": reel_offset, "min_reels": min_reels},
            "note": "Mentions mean a Reel names the concept; two Reels mentioning one concept is not a relationship."}

def neighborhood(store: Neo4jStore, kind: str, node_id: str, limit: int = 200) -> dict:
    """One node's direct neighborhood with evidence excerpts (for expansion and the inspector)."""
    if kind == "reel":
        reel = store.run(f"MATCH (r:Reel {{reel_id: $rid}}) RETURN {_reel_fields()}, r.hashtags AS hashtags",
                         rid=node_id)
        if not reel:
            return {"kind": kind, "found": False}
        rows = store.run("""
            MATCH (r:Reel {reel_id: $rid})-[:HAS_PASSAGE]->(p:Passage)-[m:MENTIONS]->(c:Concept)
            WITH c, p, m ORDER BY p.start
            WITH c, collect({chunk_id: p.chunk_id, content_type: p.content_type, start: p.start, end: p.end,
                             surface: m.surface, text: left(p.text, $ex)}) AS ev
            OPTIONAL MATCH (c)<-[:MENTIONS]-(:Passage)<-[:HAS_PASSAGE]-(o:Reel)
            RETURN c.key AS key, c.name AS name, c.type AS type, coalesce(c.aliases, []) AS aliases,
                   count(DISTINCT o) AS n_reels, size(ev) AS passages, ev[0..3] AS evidence
            ORDER BY passages DESC, name LIMIT $lim""", rid=node_id, ex=EXCERPT, lim=limit)
        return {"kind": kind, "found": True, "reel": reel[0], "concepts": rows,
                "mentions": [{"reel_id": node_id, "concept_key": c["key"], "passages": c["passages"]} for c in rows],
                "assertions": _assertions(store, "a.reel_id = $rid", rid=node_id)}
    if kind == "concept":
        c = resolve(store, node_id)
        if not c:
            return {"kind": kind, "found": False}
        n = store.run("MATCH (c:Concept {key: $k})<-[:MENTIONS]-(:Passage)<-[:HAS_PASSAGE]-(r:Reel) "
                      "RETURN count(DISTINCT r) AS n", k=c["key"])[0]["n"]
        reels = store.run(f"""
            MATCH (c:Concept {{key: $k}})<-[m:MENTIONS]-(p:Passage)<-[:HAS_PASSAGE]-(r:Reel)
            WITH r, p, m ORDER BY p.start
            WITH r, collect({{chunk_id: p.chunk_id, content_type: p.content_type, start: p.start, end: p.end,
                              surface: m.surface, text: left(p.text, $ex)}}) AS ev
            RETURN {_reel_fields()}, size(ev) AS passages, ev[0..3] AS evidence
            ORDER BY passages DESC, reel_id LIMIT $lim""", k=c["key"], ex=EXCERPT, lim=limit)
        assertions = _assertions(store, "(a.subject_key = $k OR a.object_key = $k)", k=c["key"])
        other = sorted({k for a in assertions for k in (a["subject_key"], a["object_key"])} - {c["key"]})
        return {"kind": kind, "found": True, "concept": {**c, "aliases": c.get("aliases") or [], "n_reels": n},
                "reels": reels, "related_concepts": _concept_rows(store, other),
                "mentions": [{"reel_id": r["reel_id"], "concept_key": c["key"], "passages": r["passages"]} for r in reels],
                "assertions": assertions}
    raise ValueError("kind must be 'reel' or 'concept'")

def search(store: Neo4jStore, q: str, limit: int = 20) -> dict:
    """Case-insensitive search: concept names/aliases (prefix matches first, then by Reel count) and Reels by
    owner or caption."""
    q = (q or "").strip()
    if not q:
        return {"query": q, "concepts": [], "reels": []}
    concepts = store.run("""
        MATCH (c:Concept)
        WHERE toLower(c.name) CONTAINS toLower($q) OR any(a IN coalesce(c.aliases, []) WHERE toLower(a) CONTAINS toLower($q))
        OPTIONAL MATCH (c)<-[:MENTIONS]-(:Passage)<-[:HAS_PASSAGE]-(r:Reel)
        WITH c, count(DISTINCT r) AS n_reels,
             CASE WHEN toLower(c.name) = toLower($q) THEN 0 WHEN toLower(c.name) STARTS WITH toLower($q) THEN 1
                  ELSE 2 END AS rank
        RETURN c.key AS key, c.name AS name, c.type AS type, n_reels
        ORDER BY rank, n_reels DESC, name LIMIT $lim""", q=q, lim=limit)
    reels = store.run(f"""
        MATCH (r:Reel)
        WHERE toLower(coalesce(r.owner, '')) CONTAINS toLower($q) OR toLower(coalesce(r.caption_short, '')) CONTAINS toLower($q)
        RETURN {_reel_fields()} ORDER BY r.saved_at DESC LIMIT $lim""", q=q, lim=max(1, limit // 2))
    return {"query": q, "concepts": concepts, "reels": reels}