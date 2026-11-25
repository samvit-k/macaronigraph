"""LIVE: rag.GraphReader against a THROWAWAY Neo4j (same fixture as test_graph_neo4j_live: temp dirs, random
ports, never the production database). Graph built by the real sync with the fake LLM over a temp Chroma index."""
import pytest

from backend import config, embedder, indexer, jobs, rag
from backend.graph import sync as gs
from test_graph_neo4j_live import neo, pytestmark  # noqa: F401  (fixture + skip marker)
from test_graph_sync import FakeLLM, triton_reel
from test_indexer import add

@pytest.fixture(autouse=True)
def clean(neo, isolated_output, monkeypatch):  # noqa: F811
    neo.run("MATCH (n) DETACH DELETE n")
    monkeypatch.setattr(gs, "extractor_version", lambda model_id=None: FakeLLM.version)
    monkeypatch.setattr(gs, "compatible_versions", lambda model_id=None: [])
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    jobs._export_state.update(pending=0, last=0.0)
    yield
    indexer.release_writer()

def build(neo):  # noqa: F811
    triton_reel("A")
    triton_reel("B", extra="for GPU kernels")
    add("C", "asyncio in Python with CUDA")
    indexer.sync_all()
    assert gs.sync(store=neo, llm=FakeLLM())["ready"] == 3

def test_reader_queries_and_corroboration_from_real_evidence(neo):  # noqa: F811
    build(neo)
    g = rag.GraphReader(neo)
    idx = g.concept_index()
    assert idx["n_reels"] == 3 and idx["concepts"]["cuda"]["df"] == 3 and idx["concepts"]["triton"]["df"] == 2
    rels = g.relations_for_concepts(["triton"])
    assert rels and all(r["predicate"] == "DEPENDS_ON" and r["ready"] for r in rels)
    assert {s["reel_id"] for r in rels for s in r["support"]} == {"A", "B"}
    r = rag.retrieve("Triton is built on top of CUDA", mode="hybrid", graph=g,
                     options={"rel_abs": 0.0, "rel_frac": 0.0})
    assert r["mode_used"] == "hybrid" and r["connections"]["relations"]
    assert {x["status"] for x in r["connections"]["relations"]} == {"corroborated"}   # 2 Reels, valid evidence

def test_rejected_relations_never_reach_retrieval(neo):  # noqa: F811
    build(neo)
    for a in neo.run("MATCH (a:Assertion) RETURN a.subject_key AS s, a.predicate AS p, a.object_key AS o, "
                     "a.chunk_id AS c"):
        gs.put_review(a["s"], a["p"], a["o"], a["c"], "reject", "test")
    gs.sync(store=neo, llm=FakeLLM())
    g = rag.GraphReader(neo)
    assert g.relations_for_concepts(["triton", "cuda"]) == []
    r = rag.retrieve("Triton is built on top of CUDA", mode="graph", graph=g)
    assert r["connections"]["relations"] == [] and all(ch["type"] != "relation" for s in r["sources"]
                                                      for ch in s["channels"])

def test_listing_and_validity_with_the_real_reader(neo):  # noqa: F811
    build(neo)
    g = rag.GraphReader(neo)
    r = rag.retrieve("Which Reels mention CUDA?", mode="hybrid", graph=g)
    m = r["connections"]["mentions"][0]
    assert r["intent"] == "listing" and set(m["reels"]) == {"A", "B", "C"} and m["complete"]
    neo.run("MATCH (r:Reel {reel_id: 'C'}) SET r.content_hash = 'outdated'")       # graph behind the index
    r = rag.retrieve("Which Reels mention CUDA?", mode="hybrid", graph=g)
    assert "C" not in r["connections"]["mentions"][0]["reels"] and r["stats"]["dropped_not_valid"] >= 1