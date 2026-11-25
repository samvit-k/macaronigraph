"""Retrieval over the saved Reels: vector-only, graph-only and hybrid modes with weighted rank fusion."""
from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import embedder, indexer
from .graph import normalize as nz

MODES = ("vector", "graph", "hybrid")
STATUS_WEIGHT = {"verified": 1.0, "corroborated": 0.7, "proposed": 0.4}
TEXT_KINDS = {"transcript": 1.0, "caption": 1.0, "ocr_screen": 0.8, "ocr": 0.8, "ocr_title": 0.6, "ocr_fragment": 0.3}
_LISTING = re.compile(r"^\s*(?:which|what|list|show me)\s+(?:of\s+(?:my\s+)?(?:saved\s+)?)?(?:other\s+)?(?:my\s+)?"
                      r"(?:saved\s+)?(?:reels?|videos?)\s+(?:also\s+)?(?:mention|mentions|mentioned|talk about|"
                      r"talks about|discuss|discusses|recommend|recommends|cover|covers|feature|features|"
                      r"use|uses|teach|teaches)\s+(?P<tail>.+?)\s*\??\s*$", re.I)
_ALLOWED_TAIL = frozenset("and in what which context contexts how way ways this it them that".split())

DEFAULTS = {
    "vector_candidates": 40,
    "seed_passages": 8,
    "min_specificity": 0.75,
    "co_mention": True,
    "relations": True,
    "rel_abs": 0.35,
    "rel_frac": 0.6,
    "per_concept_reels": 3,
    "boost_weight": 0.8,
    "w_vector": 1.0, "w_co_mention": 1.0, "w_relation": 0.6,
    "rrf_k": 60,
    "protect_top": 2,
    "reserve_graph": 2,
    "max_sources": 12, "per_reel": 3, "context_chars": 9000,
    "listing_max": 30,
    "near_duplicate": 0.8,
}

class RetrievalUnavailable(RuntimeError):
    pass

class _GraphFailed(Exception):
    pass

class GraphReader:

    def __init__(self, store=None):
        from .graph.store import Neo4jStore
        self.store = store or Neo4jStore()

    def concept_index(self) -> dict:
        rows = self.store.run("""MATCH (c:Concept) OPTIONAL MATCH (c)<-[:MENTIONS]-(p:Passage)
                                 RETURN c.key AS key, c.name AS name, c.type AS type, c.aliases AS aliases,
                                        count(DISTINCT p.reel_id) AS df""")
        n = self.store.run("MATCH (r:Reel) RETURN count(r) AS n")[0]["n"]
        return {"n_reels": n, "concepts": {r["key"]: {"name": r["name"], "type": r["type"], "df": r["df"],
                                                      "aliases": r["aliases"] or []} for r in rows}}

    def concepts_for_chunks(self, chunk_ids: list[str]) -> list[dict]:
        return self.store.run("""MATCH (r:Reel)-[:HAS_PASSAGE]->(p:Passage)-[:MENTIONS]->(c:Concept)
                                 WHERE p.chunk_id IN $ids
                                 RETURN p.chunk_id AS chunk_id, p.reel_id AS reel_id, c.key AS key,
                                        r.content_hash AS content_hash, r.retrieval_ready AS ready""",
                              ids=list(chunk_ids))

    def passages_for_concepts(self, keys: list[str]) -> list[dict]:
        return self.store.run("""MATCH (r:Reel)-[:HAS_PASSAGE]->(p:Passage)-[m:MENTIONS]->(c:Concept)
                                 WHERE c.key IN $keys AND p.evidence
                                 RETURN c.key AS key, p.chunk_id AS chunk_id, p.reel_id AS reel_id,
                                        p.content_type AS content_type, m.method AS method,
                                        r.content_hash AS content_hash, r.retrieval_ready AS ready""",
                              keys=list(keys))

    def relations_for_concepts(self, keys: list[str]) -> list[dict]:
        return self.store.run("""
            MATCH (s:Concept)<-[:SUBJECT]-(a:Assertion)-[:OBJECT]->(o:Concept), (a)-[:EVIDENCE]->(p:Passage),
                  (r:Reel {reel_id: a.reel_id})
            WHERE (s.key IN $keys OR o.key IN $keys) AND a.status <> 'rejected'
            OPTIONAL MATCH (b:Assertion {subject_key: a.subject_key, predicate: a.predicate, object_key: a.object_key})
                           -[:EVIDENCE]->(bp:Passage), (br:Reel {reel_id: b.reel_id})
            WHERE b.status <> 'rejected'
            RETURN a.id AS assertion_id, a.status AS stored_status, s.key AS subject, s.name AS subject_name,
                   a.predicate AS predicate, o.key AS object, o.name AS object_name, a.quote AS quote,
                   a.reel_id AS reel_id, p.chunk_id AS chunk_id, r.content_hash AS content_hash,
                   r.retrieval_ready AS ready,
                   collect(DISTINCT {reel_id: b.reel_id, chunk_id: bp.chunk_id, content_hash: br.content_hash,
                                     ready: br.retrieval_ready}) AS support""", keys=list(keys))

@dataclass
class Cand:
    chunk_id: str
    reel_id: str
    channels: list = field(default_factory=list)
    ranks: dict = field(default_factory=dict)
    similarity: float | None = None
    meta: dict = field(default_factory=dict)
    text: str = ""
    listing: bool = False

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def _words(s: str) -> list[str]:
    return re.findall(r"[\w][\w+#.\-']*[\w+#]|[\w]", s)

def _link(text: str, index: dict) -> list[tuple[str, int, int]]:
    surf: dict[str, str] = {}
    for key, c in index["concepts"].items():
        for s in [c["name"], *c.get("aliases", [])]:
            surf.setdefault(nz.base_key(s), key)
    words = _words(text)
    taken: set[int] = set()
    found: list[tuple[str, int, int]] = []
    for n in range(5, 0, -1):
        for i in range(len(words) - n + 1):
            if taken & set(range(i, i + n)):
                continue
            phrase = " ".join(words[i:i + n])
            if nz.is_generic(phrase):
                continue
            spec = nz.context_alias(phrase)
            if spec is not None:
                key = nz.base_key(spec["canonical"]) if nz.context_supports(spec, text) else nz.base_key(phrase)
            else:
                k = nz.canonicalize(phrase)[0]
                key = k if k in index["concepts"] else surf.get(nz.base_key(phrase))
            if key and key in index["concepts"]:
                taken |= set(range(i, i + n))
                found.append((key, i, i + n))
    return found

def link_concepts(question: str, index: dict) -> list[str]:
    out: list[str] = []
    for k, _, _ in sorted(_link(question, index), key=lambda x: x[1]):
        if k not in out:
            out.append(k)
    return out

def listing_target(question: str, index: dict) -> list[str]:
    m = _LISTING.match(question)
    if not m:
        return []
    tail = m.group("tail")
    links = _link(tail, index)
    if not links:
        return []
    words = _words(tail)
    covered = {i for _, a, b in links for i in range(a, b)}
    rest = [w.lower() for i, w in enumerate(words) if i not in covered]
    if any(w not in _ALLOWED_TAIL for w in rest):
        return []
    return [k for k, _, _ in links]

def specificity(df: int, n: int) -> float:
    if n <= 1 or df <= 0:
        return 1.0
    return max(0.0, math.log(n / df) / math.log(n))

def _cos(q: np.ndarray, v) -> float:
    v = np.asarray(v, dtype=np.float32)
    d = float(np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
    return float(np.dot(q, v) / d)

def retrieve(question: str, mode: str = "hybrid", options: dict | None = None, graph=None,
             boost_concepts: list[str] | None = None, exclude_reels: list[str] | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    p = {**DEFAULTS, **(options or {})}
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    fallback: list[dict] = []
    exclude = set(exclude_reels or [])
    stats = {"vector_candidates": 0, "graph_candidates": 0, "dropped_not_valid": 0, "dropped_irrelevant": 0,
             "dropped_unspecific": 0, "dropped_near_duplicate": 0, "dropped_per_concept_cap": 0}
    empty_conn = {"relations": [], "co_mentions": [], "mentions": []}

    with indexer.reading() as col_name:
        if not col_name:
            raise RetrievalUnavailable("no semantic index has been built yet; passages cannot be validated")
        col = indexer.get_collection(col_name)
        emb = embedder.for_spec(indexer.spec_of(col))
        cands: dict[str, Cand] = {}
        state = {"qv": None, "best": 0.0}

        def add(cid, rid, channel, rank, info, sim=None) -> Cand:
            c = cands.setdefault(cid, Cand(cid, rid))
            if channel not in c.ranks or rank < c.ranks[channel]:
                c.ranks[channel] = rank
            c.channels.append({"type": channel, **info})
            if sim is not None:
                c.similarity = sim if c.similarity is None else max(c.similarity, sim)
            return c

        def vector_stage() -> None:
            t = time.perf_counter()
            state["qv"] = emb.encode_query(question)
            timing["embed_s"] = time.perf_counter() - t
            t = time.perf_counter()
            res = col.query(query_embeddings=[state["qv"].tolist()],
                            n_results=max(1, min(col.count(), p["vector_candidates"])),
                            where={"source": "instagram"}, include=["documents", "metadatas", "distances"])
            hits = sorted((indexer._hit(*x) for x in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                                       res["distances"][0])), key=lambda h: -h["score"])
            hits = [h for h in hits if h["reel_id"] not in exclude]
            for r, h in enumerate(hits):
                c = add(h["chunk_id"], h["reel_id"], "semantic", r, {"score": h["score"]}, sim=h["similarity"])
                c.text, c.meta = h["text"], {**h["_reel"], **{k: h[k] for k in ("content_type", "start", "end")}}
            state["best"] = hits[0]["similarity"] if hits else 0.0
            stats["vector_candidates"] = len(hits)
            timing["vector_s"] = time.perf_counter() - t

        mode_used = mode
        if mode in ("vector", "hybrid"):
            vector_stage()

        concepts_out: dict[str, dict] = {}
        connections = copy.deepcopy(empty_conn)
        mentions_of: dict[str, set[str]] = {}
        listed: dict[str, list[str]] = {}
        names: dict[str, str] = {}
        if mode in ("graph", "hybrid"):
            t = time.perf_counter()
            snapshot = copy.deepcopy(cands)
            stats_before = dict(stats)
            try:
                concepts_out, connections, mentions_of, listed, names = _graph_stage(
                    question, mode, p, graph, col, state, exclude, cands, add, stats, boost_concepts or [])
            except _GraphFailed as e:
                cands.clear()
                cands.update(snapshot)
                stats.clear()
                stats.update(stats_before)
                concepts_out, connections, mentions_of, listed, names = {}, copy.deepcopy(empty_conn), {}, {}, {}
                fallback.append({"from": mode, "to": "vector", "reason": f"graph unavailable: {e}"})
                mode_used = "vector"
                if mode == "graph":
                    vector_stage()
            timing["graph_s"] = time.perf_counter() - t

        t = time.perf_counter()
        w = {"semantic": p["w_vector"], "shared_concept": p["w_co_mention"], "relation": p["w_relation"]}
        for c in cands.values():
            c.meta["fused"] = sum(w[ch] / (p["rrf_k"] + r + 1) for ch, r in c.ranks.items())
        ordered = sorted(cands.values(), key=lambda c: -c.meta["fused"])
        protected: list[str] = []
        if mode_used == "hybrid" and p["protect_top"]:
            for c in sorted((c for c in cands.values() if "semantic" in c.ranks), key=lambda c: c.ranks["semantic"]):
                if c.reel_id not in protected:
                    protected.append(c.reel_id)
            protected = protected[:p["protect_top"]]
            ordered.sort(key=lambda c: protected.index(c.reel_id) if c.reel_id in protected else len(protected))
        reels: list[str] = []
        for c in ordered:
            if c.reel_id not in reels:
                reels.append(c.reel_id)

        n_listed = len({r for rs in listed.values() for r in rs})
        sources = _pack(ordered, p, n_listed, protected, stats)
        in_pack = {s["chunk_id"] for s in sources}
        pack_reels: dict[str, set[str]] = {}
        for s in sources:
            for k in mentions_of.get(s["chunk_id"], ()):
                pack_reels.setdefault(k, set()).add(s["reel_id"])
        connections["co_mentions"] = [
            {"concept": names.get(k, k), "key": k, "reels": sorted(rs),
             "note": "these Reels each mention the concept; this is not a stated relationship between them"}
            for k, rs in sorted(pack_reels.items()) if len(rs) >= 2]
        connections["relations"] = [r for r in connections["relations"] if r["chunk_id"] in in_pack]
        for k, rs in listed.items():
            covered = sorted(pack_reels.get(k, set()) & set(rs))
            connections["mentions"].append({"concept": names.get(k, k), "key": k,
                                            "reels": rs, "in_pack": covered, "complete": set(covered) == set(rs)})
        timing["rank_s"] = time.perf_counter() - t
    timing["total_s"] = time.perf_counter() - t0
    source_concepts = {s["chunk_id"]: [{"key": k, "name": names.get(k, k)} for k in sorted(mentions_of.get(s["chunk_id"], ()))]
                       for s in sources}
    return {"question": question, "mode_requested": mode, "mode_used": mode_used, "fallback": fallback,
            "intent": "listing" if listed else "question", "collection": col_name, "sources": sources,
            "source_concepts": source_concepts, "reels": reels,
            "concepts": sorted(concepts_out.values(), key=lambda c: (not c["named_in_question"], -c["specificity"])),
            "connections": connections, "timing": {k: round(v, 4) for k, v in timing.items()}, "stats": stats,
            "params": p}

def _graph_stage(question, mode, p, graph, col, state, exclude, cands, add, stats, boost):
    try:
        graph = graph or GraphReader()
        index = graph.concept_index()
        n = max(1, index["n_reels"])
        linked = link_concepts(question, index)
        listing_keys = listing_target(question, index)
        qv, best = state["qv"], state["best"]
        cache: dict[str, tuple | None] = {}

        def fetch(ids):
            need = sorted({i for i in ids if i and i not in cache})
            if need:
                got = col.get(ids=need, include=["documents", "metadatas", "embeddings"])
                embs = got["embeddings"] if got["embeddings"] is not None else [None] * len(got["ids"])
                for i, d, m, e in zip(got["ids"], got["documents"], got["metadatas"], embs):
                    cache[i] = (d, m or {}, e)
                for i in need:
                    cache.setdefault(i, None)

        def valid(chunk_id, content_hash, ready) -> bool:
            v = cache.get(chunk_id)
            return bool(ready) and v is not None and v[1].get("content_hash") == content_hash

        seeds: dict[str, float] = {k: 1.0 for k in linked}
        for k in boost:
            if k in index["concepts"]:
                seeds.setdefault(k, p["boost_weight"])
        if mode == "hybrid":
            top = sorted(cands.values(), key=lambda c: c.ranks.get("semantic", 1e9))[:p["seed_passages"]]
            sim_of = {c.chunk_id: c.similarity or 0.0 for c in top}
            rows = graph.concepts_for_chunks(list(sim_of))
            fetch(r["chunk_id"] for r in rows)
            for r in rows:
                if valid(r["chunk_id"], r["content_hash"], r["ready"]):
                    seeds[r["key"]] = max(seeds.get(r["key"], 0.0), sim_of[r["chunk_id"]])
                else:
                    stats["dropped_not_valid"] += 1

        concepts_out, expand = {}, []
        for k in seeds:
            c = index["concepts"].get(k)
            if c is None:
                continue
            sp = specificity(c["df"], n)
            concepts_out[k] = {"key": k, "name": c["name"], "type": c["type"], "df": c["df"],
                               "specificity": round(sp, 3), "seed_weight": round(seeds[k], 4),
                               "named_in_question": k in linked,
                               "from_conversation": k in boost and k not in linked, "expanded": False}
            if k in linked or sp >= p["min_specificity"]:
                expand.append(k)
                concepts_out[k]["expanded"] = True
            else:
                stats["dropped_unspecific"] += 1

        rows = []
        if expand and (mode == "graph" or p["co_mention"] or listing_keys):
            rows += [("shared_concept", r) for r in graph.passages_for_concepts(expand)]
        rel_rows = graph.relations_for_concepts(expand) if expand and (mode == "graph" or p["relations"]) else []
        rows += [("relation", r) for r in rel_rows]
        rows = [(ch, r) for ch, r in rows if r["reel_id"] not in exclude]
        stats["graph_candidates"] = len(rows)
        fetch([r["chunk_id"] for _, r in rows] +
              [s.get("chunk_id") for ch, r in rows if ch == "relation" for s in r["support"]])

        connections = {"relations": [], "co_mentions": [], "mentions": []}
        listed: dict[str, list[str]] = {k: [] for k in listing_keys}
        scored: dict[str, list] = {"shared_concept": [], "relation": []}
        for ch, r in rows:
            if not valid(r["chunk_id"], r["content_hash"], r["ready"]):
                stats["dropped_not_valid"] += 1
                continue
            d, md, e = cache[r["chunk_id"]]
            if ch == "relation":
                support = {s["reel_id"] for s in r["support"]
                           if s.get("chunk_id") and valid(s["chunk_id"], s["content_hash"], s["ready"])}
                r = {**r, "status": "verified" if r["stored_status"] == "verified"
                     else "corroborated" if len(support) >= 2 else "proposed", "support": len(support)}
                key = r["subject"] if r["subject"] in expand else r["object"]
            else:
                key = r["key"]
            sp = concepts_out.get(key, {}).get("specificity", 1.0)
            status_w = STATUS_WEIGHT[r["status"]] if ch == "relation" else 1.0
            exhaustive = ch == "shared_concept" and key in listed
            if exhaustive and r["reel_id"] not in listed[key]:
                listed[key].append(r["reel_id"])
            if mode == "hybrid" and not exhaustive:
                sim = _cos(qv, e)
                if sim < p["rel_abs"] or sim < p["rel_frac"] * best:
                    stats["dropped_irrelevant"] += 1
                    continue
                weight = sim * (0.5 + 0.5 * sp) * seeds.get(key, 0.0) * status_w
            else:
                sim = _cos(qv, e) if qv is not None else None
                quality = 1.0 if (r.get("method") or "llm").startswith("llm") else 0.8
                weight = sp * quality * TEXT_KINDS.get(md.get("content_type"), 0.5) * seeds.get(key, 0.0) * status_w
            scored[ch].append((weight, sim, key, r, (d, md), exhaustive))

        for ch, items in scored.items():
            items.sort(key=lambda x: -x[0])
            per_concept: dict[str, set] = {}
            rank = 0
            for weight, sim, key, r, (d, md), exhaustive in items:
                reels = per_concept.setdefault(key, set())
                if (ch == "shared_concept" and not exhaustive and r["reel_id"] not in reels
                        and len(reels) >= p["per_concept_reels"] and r["chunk_id"] not in cands):
                    stats["dropped_per_concept_cap"] += 1
                    continue
                reels.add(r["reel_id"])
                info = ({"concept": key, "concept_name": index["concepts"][key]["name"],
                         "specificity": concepts_out.get(key, {}).get("specificity"),
                         "note": "shared concept mention, not a stated relationship"}
                        if ch == "shared_concept" else
                        {"assertion_id": r["assertion_id"], "status": r["status"], "support_reels": r["support"],
                         "subject": r["subject_name"], "predicate": r["predicate"], "object": r["object_name"]})
                c = add(r["chunk_id"], r["reel_id"], ch, rank, info, sim=sim)
                c.listing = c.listing or exhaustive
                rank += 1
                if not c.text:
                    c.text = d
                    c.meta = {"source": md.get("source"), "source_url": md.get("source_url"),
                              "owner_username": md.get("owner_username"), "caption_short": md.get("caption_short"),
                              "content_type": md.get("content_type"), "start": md.get("start"), "end": md.get("end")}
                if ch == "relation" and not any(x["assertion_id"] == r["assertion_id"] for x in connections["relations"]):
                    connections["relations"].append({
                        "assertion_id": r["assertion_id"], "subject": r["subject_name"], "predicate": r["predicate"],
                        "object": r["object_name"], "status": r["status"], "support_reels": r["support"],
                        "reel_id": r["reel_id"], "chunk_id": r["chunk_id"], "quote": r["quote"],
                        "note": {"proposed": "model-proposed, unverified: tentative evidence only",
                                 "corroborated": "proposed independently in 2+ Reels with valid evidence; not verified",
                                 "verified": "human-verified"}[r["status"]]})

        ids = sorted(cands)
        fetch(ids)
        mentions_of: dict[str, set[str]] = {}
        for r in graph.concepts_for_chunks(ids):
            if valid(r["chunk_id"], r["content_hash"], r["ready"]):
                mentions_of.setdefault(r["chunk_id"], set()).add(r["key"])
        names = {k: c["name"] for k, c in index["concepts"].items()}
        return concepts_out, connections, mentions_of, listed, names
    except Exception as e:  # noqa: BLE001
        raise _GraphFailed(f"{type(e).__name__}: {e}") from e

def _pack(ordered: list[Cand], p: dict, n_listed: int, protected: list[str], stats: dict) -> list[dict]:
    listing = n_listed > 0
    limit = p["max_sources"] + min(n_listed, p["listing_max"])
    budget = p["context_chars"] * (3 if listing else 1)
    chosen: list[Cand] = []
    per_reel: dict[str, int] = {}
    toks_of: dict[str, list] = {}
    used = {"chars": 0}
    duplicates: set[int] = set()

    def take(c: Cand, cap: int, char_limit: int) -> bool:
        if any(x is c for x in chosen) or per_reel.get(c.reel_id, 0) >= cap:
            return False
        toks = _tokens(c.text)
        if any(len(toks & o) / max(1, len(toks | o)) >= p["near_duplicate"] for o in toks_of.get(c.reel_id, [])):
            duplicates.add(id(c))
            return False
        if chosen and used["chars"] + len(c.text) > char_limit:
            return False
        chosen.append(c)
        per_reel[c.reel_id] = per_reel.get(c.reel_id, 0) + 1
        toks_of.setdefault(c.reel_id, []).append(toks)
        used["chars"] += len(c.text)
        return True

    if listing:
        for c in ordered:
            if c.listing and len(chosen) < limit:
                take(c, 1, budget)
    complementary = [c for c in ordered if {"shared_concept", "relation"} & set(c.ranks)
                     and c.reel_id not in protected and not any(x is c for x in chosen)]
    reserve = min(p["reserve_graph"], len(complementary))
    reserve_chars = min(budget // 2, sum(len(c.text) for c in complementary[:reserve]))
    for c in ordered:
        if len(chosen) >= limit - reserve:
            break
        take(c, p["per_reel"], budget - reserve_chars)
    for c in complementary:
        if len(chosen) >= limit:
            break
        take(c, p["per_reel"], budget)
    for c in ordered:
        if len(chosen) >= limit:
            break
        take(c, p["per_reel"], budget)
    stats["dropped_near_duplicate"] += len(duplicates)
    rank = {id(c): i for i, c in enumerate(ordered)}
    chosen.sort(key=lambda c: rank[id(c)])
    return [{"sid": f"S{i + 1}", "chunk_id": c.chunk_id, "reel_id": c.reel_id,
             "source_url": c.meta.get("source_url") or f"https://www.instagram.com/reel/{c.reel_id}/",
             "owner": c.meta.get("owner_username"), "content_type": c.meta.get("content_type"),
             "start": c.meta.get("start"), "end": c.meta.get("end"), "text": c.text,
             "similarity": None if c.similarity is None else round(c.similarity, 4),
             "score": round(c.meta["fused"], 5), "channels": _merge_channels(c.channels)}
            for i, c in enumerate(chosen)]

def _merge_channels(chs: list[dict]) -> list[dict]:
    out, seen = [], set()
    for ch in chs:
        key = (ch["type"], ch.get("concept") or ch.get("assertion_id"))
        if key not in seen:
            seen.add(key)
            out.append(ch)
    return out