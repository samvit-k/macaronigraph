"""Retrieval with a mocked graph over a real temporary Chroma index (hashing embedder)."""
import json
import re
import threading
import time

import pytest

from backend import config, embedder, indexer, jobs, models, rag
from test_chunking import oseg
from test_indexer import add

@pytest.fixture(autouse=True)
def env(isolated_output, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    jobs._export_state.update(pending=0, last=0.0)
    yield
    indexer.release_writer()

TERMS = {"triton": "Triton", "cuda": "CUDA", "github": "GitHub", "python": "Python", "sourdough": "sourdough",
         "kubernetes": "Kubernetes"}

class FakeGraph:

    def __init__(self, relations=(), stale=(), not_ready=(), fail=None):
        col = indexer.get_collection(indexer.active_collection())
        got = col.get(include=["documents", "metadatas"])
        self.chunks = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
        self.stale, self.not_ready, self.fail = set(stale), set(not_ready), fail
        self.relations, self.calls = list(relations), {}

    def _hit(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.fail and self.fail[0] == name and self.calls[name] == self.fail[1]:
            raise ConnectionError(f"neo4j dropped during {name}")

    def _hash(self, rid):
        return "stale-hash" if rid in self.stale else next(m["content_hash"] for d, m in self.chunks.values()
                                                            if m["reel_id"] == rid)

    def _mentions(self, cid):
        d, _ = self.chunks[cid]
        return [k for k, w in TERMS.items() if re.search(rf"\b{w}\b", d, re.I)]

    def concept_index(self):
        self._hit("concept_index")
        reels = {m["reel_id"] for _, m in self.chunks.values()}
        df = {k: len({self.chunks[c][1]["reel_id"] for c in self.chunks if k in self._mentions(c)}) for k in TERMS}
        return {"n_reels": len(reels), "concepts": {k: {"name": TERMS[k], "type": "technology", "df": df[k],
                                                        "aliases": []} for k in TERMS if df[k]}}

    def concepts_for_chunks(self, ids):
        self._hit("concepts_for_chunks")
        return [{"chunk_id": c, "reel_id": self.chunks[c][1]["reel_id"], "key": k,
                 "content_hash": self._hash(self.chunks[c][1]["reel_id"]),
                 "ready": self.chunks[c][1]["reel_id"] not in self.not_ready}
                for c in ids if c in self.chunks for k in self._mentions(c)]

    def passages_for_concepts(self, keys):
        self._hit("passages_for_concepts")
        return [{"key": k, "chunk_id": c, "reel_id": m["reel_id"], "content_type": m["content_type"],
                 "method": "llm+exact_match", "content_hash": self._hash(m["reel_id"]),
                 "ready": m["reel_id"] not in self.not_ready}
                for c, (d, m) in self.chunks.items() for k in self._mentions(c) if k in keys]

    def relations_for_concepts(self, keys):
        self._hit("relations_for_concepts")
        return [r for r in self.relations if (r["subject"] in keys or r["object"] in keys)
                and r["stored_status"] != "rejected"]

def chunk_of(rid, kind="transcript"):
    return f"{rid}:{kind}:000"

def relation(rid, s, o, support, status="proposed", chunk=None, rel_id=None):
    chunk = chunk or chunk_of(rid)
    return {"assertion_id": rel_id or f"{rid}-{s}-{o}", "stored_status": status, "subject": s,
            "subject_name": TERMS[s], "predicate": "DEPENDS_ON", "object": o, "object_name": TERMS[o],
            "quote": "Triton is built on top of CUDA", "reel_id": rid, "chunk_id": chunk, "content_hash": None,
            "ready": True, "support": support}

def fix_hashes(g: FakeGraph, rels):
    for r in rels:
        r["content_hash"] = g._hash(r["reel_id"])
        for s in r["support"]:
            s.setdefault("content_hash", g._hash(s["reel_id"]))
            s.setdefault("ready", True)
    g.relations = rels
    return g

def corpus():
    add("A", "Triton writes fast GPU kernels in Python")
    add("B", "Triton compiler tuning makes GPU kernels faster")
    add("C", "Triton sourdough bread starter recipe")
    add("D", "Python asyncio event loops explained")
    indexer.sync_all()

def test_vector_mode_matches_phase3_search():
    corpus()
    r = rag.retrieve("fast GPU kernels with Triton", mode="vector")
    assert r["mode_used"] == "vector" and not r["fallback"]
    assert r["reels"][0] == indexer.search("fast GPU kernels with Triton")["results"][0]["reel_id"]
    assert all(ch["type"] == "semantic" for s in r["sources"] for ch in s["channels"])

def test_graph_mode_uses_no_vector_similarity(monkeypatch):
    corpus()
    monkeypatch.setattr(embedder.HashEmbedder, "encode_query",
                        lambda self, t: pytest.fail("graph-only mode must not embed the query"))
    r = rag.retrieve("Tell me about Triton", mode="graph", graph=FakeGraph())
    assert r["mode_used"] == "graph" and set(r["reels"]) == {"A", "B", "C"}
    assert all(s["similarity"] is None for s in r["sources"])
    assert all(ch["type"] in ("shared_concept", "relation") for s in r["sources"] for ch in s["channels"])

def test_graph_mode_with_no_named_concept_returns_nothing():
    corpus()
    r = rag.retrieve("how do I bake bread", mode="graph", graph=FakeGraph())
    assert r["sources"] == [] and r["mode_used"] == "graph"

def test_hybrid_adds_a_relevant_cross_reel_passage_and_drops_irrelevant_ones():
    corpus()
    r = rag.retrieve("fast GPU kernels with Triton", mode="hybrid", graph=FakeGraph(),
                     options={"vector_candidates": 1, "rel_abs": 0.3, "rel_frac": 0.3, "protect_top": 1})
    by = {s["reel_id"]: s for s in r["sources"]}
    assert "B" in by and any(ch["type"] == "shared_concept" and ch["concept"] == "triton"
                             for ch in by["B"]["channels"])
    assert "C" not in by and r["stats"]["dropped_irrelevant"] >= 1
    assert all(ch["note"].startswith("shared concept mention") for s in r["sources"]
               for ch in s["channels"] if ch["type"] == "shared_concept")

def test_unspecific_concepts_expand_only_when_the_question_names_them():
    for rid, w in [("P1", "Python lists"), ("P2", "Python dicts"), ("P3", "Python sets"), ("P4", "Python loops"),
                   ("P5", "Python classes"), ("T1", "Triton kernels in Python")]:
        add(rid, w)
    indexer.sync_all()
    r = rag.retrieve("Triton kernels", mode="hybrid", graph=FakeGraph(), options={"rel_abs": 0.0, "rel_frac": 0.0})
    py = next(c for c in r["concepts"] if c["key"] == "python")
    assert py["expanded"] is False and r["stats"]["dropped_unspecific"] >= 1
    r = rag.retrieve("Python classes", mode="hybrid", graph=FakeGraph(), options={"rel_abs": 0.0, "rel_frac": 0.0})
    assert next(c for c in r["concepts"] if c["key"] == "python")["expanded"] is True

def test_specificity_threshold_scales_with_the_collection():
    assert rag.specificity(4, 59) >= 0.65 > rag.specificity(5, 59)
    assert rag.specificity(11, 1000) >= 0.65 > rag.specificity(12, 1000)

@pytest.mark.parametrize("fail", [("concept_index", 1), ("concepts_for_chunks", 1), ("passages_for_concepts", 1),
                                  ("relations_for_concepts", 1), ("concepts_for_chunks", 2)])
def test_any_graph_failure_falls_back_to_the_exact_vector_result(fail):
    corpus()
    q = "fast GPU kernels with Triton"
    vec = rag.retrieve(q, mode="vector")
    rels = [relation("A", "triton", "cuda", [{"reel_id": "A", "chunk_id": chunk_of("A")}])]
    g = fix_hashes(FakeGraph(fail=fail), rels)
    r = rag.retrieve(q, mode="hybrid", graph=g, options={"rel_abs": 0.0, "rel_frac": 0.0})
    assert r["mode_used"] == "vector" and r["fallback"][0]["reason"].startswith("graph unavailable")
    assert r["sources"] == vec["sources"] and r["reels"] == vec["reels"]
    assert r["concepts"] == [] and r["connections"] == {"relations": [], "co_mentions": [], "mentions": []}

def test_graph_mode_failure_falls_back_to_vector():
    corpus()
    r = rag.retrieve("Tell me about Triton", mode="graph", graph=FakeGraph(fail=("concept_index", 1)))
    assert r["mode_used"] == "vector" and r["sources"] and r["fallback"][0]["to"] == "vector"

@pytest.mark.parametrize("kind", ["stale", "not_ready"])
def test_invalid_graph_evidence_is_excluded_while_valid_expansion_still_happens(kind):
    corpus()
    g = FakeGraph(stale={"B"}) if kind == "stale" else FakeGraph(not_ready={"B"})
    r = rag.retrieve("fast GPU kernels with Triton", mode="hybrid", graph=g,
                     options={"vector_candidates": 1, "rel_abs": 0.0, "rel_frac": 0.0, "protect_top": 1})
    graph_reels = {s["reel_id"] for s in r["sources"] for ch in s["channels"] if ch["type"] != "semantic"}
    assert "B" not in r["reels"] and r["stats"]["dropped_not_valid"] >= 1
    assert "C" in graph_reels and graph_reels <= {"A", "C"}

def test_stale_mentions_never_seed_the_expansion():
    corpus()
    g = FakeGraph(stale={"A"})
    r = rag.retrieve("fast GPU kernels", mode="hybrid", graph=g,
                     options={"vector_candidates": 1, "rel_abs": 0.0, "rel_frac": 0.0, "protect_top": 1})
    assert r["concepts"] == [] and r["reels"] == ["A"]

def test_relations_reported_only_for_evidence_in_the_pack_and_corroboration_needs_valid_evidence():
    corpus()
    two_valid = [{"reel_id": "A", "chunk_id": chunk_of("A")}, {"reel_id": "B", "chunk_id": chunk_of("B")}]
    one_stale = [{"reel_id": "A", "chunk_id": chunk_of("A")},
                 {"reel_id": "C", "chunk_id": chunk_of("C"), "content_hash": "stale-hash", "ready": True}]
    one_unready = [{"reel_id": "A", "chunk_id": chunk_of("A")},
                   {"reel_id": "C", "chunk_id": chunk_of("C"), "ready": False}]
    opts = {"rel_abs": 0.0, "rel_frac": 0.0}
    g = fix_hashes(FakeGraph(), [relation("A", "triton", "cuda", two_valid)])
    r = rag.retrieve("Triton kernels", mode="hybrid", graph=g, options=opts)
    assert [x["status"] for x in r["connections"]["relations"]] == ["corroborated"]
    for support in (one_stale, one_unready):
        g = fix_hashes(FakeGraph(), [relation("A", "triton", "cuda", support)])
        r = rag.retrieve("Triton kernels", mode="hybrid", graph=g, options=opts)
        assert [x["status"] for x in r["connections"]["relations"]] == ["proposed"]
    g = fix_hashes(FakeGraph(), [relation("C", "triton", "cuda", [{"reel_id": "C", "chunk_id": chunk_of("C", "caption")}],
                                          chunk=chunk_of("C", "caption"))])
    r = rag.retrieve("Triton kernels", mode="hybrid", graph=g,
                     options={**opts, "max_sources": 1, "protect_top": 3})
    in_pack = {s["chunk_id"] for s in r["sources"]}
    assert chunk_of("C", "caption") not in in_pack and r["connections"]["relations"] == []

def test_co_mentions_need_two_distinct_reels_in_the_pack():
    corpus()
    r = rag.retrieve("fast GPU kernels with Triton", mode="hybrid", graph=FakeGraph(),
                     options={"rel_abs": 0.0, "rel_frac": 0.0})
    assert r["connections"]["co_mentions"]
    for cm in r["connections"]["co_mentions"]:
        assert len(set(cm["reels"])) >= 2 and "not a stated relationship" in cm["note"]
    r = rag.retrieve("fast GPU kernels with Triton", mode="hybrid", graph=FakeGraph(),
                     options={"rel_abs": 0.0, "rel_frac": 0.0, "max_sources": 1})
    assert r["connections"]["co_mentions"] == []

def _github_corpus():
    for i in range(6):
        add(f"G{i}", f"GitHub repo number {i} for side projects")
    add("X", "Unrelated sourdough recipe")
    indexer.sync_all()

def test_listing_questions_are_exhaustive_without_the_per_concept_cap():
    _github_corpus()
    r = rag.retrieve("Which Reels mention GitHub?", mode="hybrid", graph=FakeGraph(),
                     options={"per_concept_reels": 2})
    m = r["connections"]["mentions"][0]
    assert r["intent"] == "listing" and set(m["reels"]) == {f"G{i}" for i in range(6)}
    assert m["complete"] is True and set(m["in_pack"]) == set(m["reels"])
    assert len(r["sources"]) <= rag.DEFAULTS["max_sources"] + 6
    names = {c["concept"] for c in r["connections"]["co_mentions"]}
    assert names <= set(TERMS.values())
    r = rag.retrieve("Which of my saved Reels mention GitHub, and in what context?", mode="graph", graph=FakeGraph())
    assert r["intent"] == "listing" and len(r["connections"]["mentions"][0]["reels"]) == 6

def test_listing_completeness_reflects_the_final_pack():
    _github_corpus()
    r = rag.retrieve("Which Reels mention GitHub?", mode="hybrid", graph=FakeGraph(),
                     options={"listing_max": 3, "max_sources": 0})
    m = r["connections"]["mentions"][0]
    assert len(m["reels"]) == 6 and len(m["in_pack"]) == 3 and m["complete"] is False

def test_constrained_questions_are_not_exhaustive_listings():
    add("I1", "Python for coding interviews with NeetCode")
    add("I2", "Python for CUDA kernels")
    add("I3", "Python asyncio")
    indexer.sync_all()
    idx = FakeGraph().concept_index()
    assert rag.listing_target("Which Reel teaches Python specifically for coding interviews?", idx) == []
    assert rag.listing_target("Which Reels mention Python?", idx) == ["python"]
    assert rag.listing_target("Which other Reels mention Python?", idx) == ["python"]
    r = rag.retrieve("Which Reel teaches Python specifically for coding interviews?", mode="hybrid", graph=FakeGraph())
    assert r["intent"] == "question" and r["connections"]["mentions"] == []

def _boost_corpus():
    corpus()
    for i in range(21):
        add(f"F{i}", f"gardening tomatoes week {i}")
    indexer.sync_all()

def test_boost_concepts_seed_expansion_with_the_same_gates():
    _boost_corpus()
    opts = {"vector_candidates": 1, "protect_top": 1, "rel_abs": 0.3, "rel_frac": 0.3, "seed_passages": 0,
            "min_specificity": 0.65}
    idx = FakeGraph().concept_index()
    assert rag.specificity(idx["concepts"]["triton"]["df"], idx["n_reels"]) >= opts["min_specificity"]
    r = rag.retrieve("which ones make GPU kernels faster", mode="hybrid", graph=FakeGraph(), options=opts,
                     boost_concepts=["triton"])
    t = next(c for c in r["concepts"] if c["key"] == "triton")
    assert t["from_conversation"] and t["expanded"] and t["seed_weight"] == 0.8
    assert "C" not in r["reels"]
    r = rag.retrieve("which ones make GPU kernels faster", mode="hybrid", graph=FakeGraph(), options=opts,
                     boost_concepts=[])
    assert all(c["key"] != "triton" for c in r["concepts"])

def test_unspecific_boost_is_not_expanded():
    for rid, w in [("P1", "Python lists"), ("P2", "Python dicts"), ("P3", "Python sets"), ("P4", "Python loops")]:
        add(rid, w)
    indexer.sync_all()
    r = rag.retrieve("which ones explain loops", mode="hybrid", graph=FakeGraph(), boost_concepts=["python"],
                     options={"seed_passages": 0})
    assert next(c for c in r["concepts"] if c["key"] == "python")["expanded"] is False

def test_exclude_reels_is_honoured():
    corpus()
    r = rag.retrieve("Which other Reels mention Triton?", mode="hybrid", graph=FakeGraph(), exclude_reels=["A"])
    assert "A" not in r["reels"] and set(r["connections"]["mentions"][0]["reels"]) == {"B", "C"}

def test_reserved_slots_hold_complementary_evidence_not_more_of_the_protected_reels():
    many = [oseg(i, i + 1, f"Triton kernel tip number {i} about GPU speed and tiling") for i in range(8)]
    add("A", "Triton GPU kernels speed tiling", ocr=many)
    add("B", "Triton GPU kernels speed tiling", ocr=many)
    add("E", "Triton autotuning guide")
    indexer.sync_all()
    opts = {"protect_top": 2, "per_reel": 6, "max_sources": 6, "reserve_graph": 1, "rel_abs": 0.0, "rel_frac": 0.0}
    r = rag.retrieve("Triton GPU kernels speed tiling", mode="hybrid", graph=FakeGraph(), options=opts)
    reels = [s["reel_id"] for s in r["sources"]]
    assert "E" in reels and reels.count("A") + reels.count("B") == 5
    r = rag.retrieve("Triton GPU kernels speed tiling", mode="hybrid", graph=FakeGraph(),
                     options={**opts, "context_chars": 400})
    assert "E" in [s["reel_id"] for s in r["sources"]]

def test_distinct_caption_and_transcript_are_both_kept_only_identical_text_is_deduplicated():
    add("A", "Triton kernels", extra_transcript="tiling shared memory and autotuning block sizes for speed")
    indexer.sync_all()
    r = rag.retrieve("Triton kernels tiling", mode="vector")
    kinds = {s["content_type"] for s in r["sources"] if s["reel_id"] == "A"}
    assert {"caption", "transcript"} <= kinds
    dup = rag.Cand("x:ocr:000", "A", text="Triton kernels tiling"), rag.Cand("x:ocr:001", "A", text="Triton kernels tiling")
    for c in dup:
        c.ranks, c.meta = {"semantic": 0}, {"fused": 1.0}
    stats = {"dropped_near_duplicate": 0}
    out = rag._pack(list(dup), rag.DEFAULTS, 0, [], stats)
    assert len(out) == 1 and stats["dropped_near_duplicate"] == 1

def test_retrieval_does_not_wait_for_the_heavy_lock():
    corpus()
    entered, release = threading.Event(), threading.Event()

    def hold():
        with models.heavy("answer generation"):
            entered.set()
            release.wait(10)
    t = threading.Thread(target=hold)
    t.start()
    entered.wait(5)
    try:
        t0 = time.monotonic()
        r = rag.retrieve("fast GPU kernels with Triton", mode="hybrid", graph=FakeGraph())
        assert r["sources"] and time.monotonic() - t0 < 5
    finally:
        release.set()
        t.join(5)

def test_retrieval_during_index_embedding_waits_only_for_the_embedder_and_never_deadlocks():
    corpus()
    lock = threading.RLock()
    entered, release, done = threading.Event(), threading.Event(), {}

    def embedding():
        with models.heavy("index embedding"), lock:
            entered.set()
            release.wait(10)
    t = threading.Thread(target=embedding)
    t.start()
    entered.wait(5)

    def query():
        with lock:
            done["r"] = rag.retrieve("fast GPU kernels with Triton", mode="vector")
    q = threading.Thread(target=query)
    q.start()
    time.sleep(0.3)
    assert "r" not in done
    release.set()
    t.join(5)
    q.join(5)
    assert not q.is_alive() and done["r"]["sources"]

def test_defaults_are_the_frozen_dev_tuned_parameters():
    frozen = json.loads((config.ROOT / "fixtures" / "rag_params.json").read_text())
    assert frozen["tuned_on"] == "dev" and frozen["params"] == rag.DEFAULTS