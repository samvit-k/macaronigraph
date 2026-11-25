"""Live Neo4j tests against a throwaway instance with its own directories and random localhost ports."""
import os
import subprocess
import time
from pathlib import Path

import pytest

from backend import config, embedder, indexer, jobs
from backend.graph import queries as q
from backend.graph import store as st
from backend.graph import sync as gs
from helpers import free_port, production_path_guard
from test_graph_sync import FakeLLM, triton_reel
from test_indexer import add

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "neo4j.sh"
pytestmark = pytest.mark.skipif(not (ROOT / ".tools" / "neo4j" / "bin" / "neo4j").exists(),
                                reason="Neo4j not installed (scripts/neo4j.sh install)")

@pytest.fixture(scope="module")
def neo(tmp_path_factory):
    d = tmp_path_factory.mktemp("neo4j-throwaway")
    bolt, http = free_port(), free_port()
    env = {**os.environ, "NEO4J_CONF_DIR": str(d / "conf"), "NEO4J_DATA_DIR": str(d / "data"),
           "NEO4J_ENV_FILE": str(d / ".env"), "NEO4J_BOLT_PORT": str(bolt), "NEO4J_HTTP_PORT": str(http)}
    for cmd in ("install", "start"):
        r = subprocess.run(["bash", str(SCRIPT), cmd], env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr + r.stdout
    creds = dict(line.split("=", 1) for line in (d / ".env").read_text().splitlines() if "=" in line)
    assert creds["NEO4J_URI"] == f"bolt://127.0.0.1:{bolt}"
    s = st.Neo4jStore(uri=creds["NEO4J_URI"], user=creds["NEO4J_USER"], password=creds["NEO4J_PASSWORD"])
    for _ in range(120):
        try:
            s.ping()
            break
        except st.GraphOffline:
            time.sleep(0.5)
    else:
        pytest.fail("throwaway Neo4j did not start")
    s.ensure_schema()
    yield s
    s.close()
    subprocess.run(["bash", str(SCRIPT), "stop"], env=env, capture_output=True, timeout=120)

@pytest.fixture(autouse=True)
def clean(neo, isolated_output, monkeypatch):
    neo.run("MATCH (n) DETACH DELETE n")
    monkeypatch.setattr(gs, "extractor_version", lambda model_id=None: FakeLLM.version)
    monkeypatch.setattr(gs, "compatible_versions", lambda model_id=None: [])
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    jobs._export_state.update(pending=0, last=0.0)
    with production_path_guard(isolated_output):
        yield

def doc(rid, concepts, assertions=(), passages=None, h="h1"):
    passages = passages or [{"chunk_id": f"{rid}:transcript:000", "reel_id": rid, "content_type": "transcript",
                             "text": " ".join(c[1] for c in concepts), "evidence": True}]
    pid = passages[0]["chunk_id"]
    return {"reel": {"reel_id": rid, "source": "instagram", "source_url": f"u/{rid}"}, "passages": passages,
            "concepts": [{"key": k, "name": n, "type": "technology", "curated": False} for k, n in concepts],
            "mentions": [{"chunk_id": pid, "key": k, "surface": n, "method": "test", "extractor_version": "x"}
                         for k, n in concepts],
            "assertions": [{"id": f"{rid}-{s}-{o}", "reel_id": rid, "chunk_id": pid, "subject": s, "predicate": p,
                            "object": o, "quote": "q", "method": "test", "confidence": "c", "provenance": "quote_verified",
                            "status": "proposed", "review_note": None, "extractor_version": "x"}
                           for s, p, o in assertions],
            "graph_hash": h, "retrieval_ready": True}

def keys(neo):
    return {r["k"] for r in neo.run("MATCH (c:Concept) RETURN c.key AS k")}

def test_constraints_exist_and_reject_duplicates(neo):
    names = {r["name"] for r in neo.run("SHOW CONSTRAINTS YIELD name RETURN name")}
    assert {"reel_id", "passage_id", "concept_key", "assertion_id"} <= names
    neo.run("CREATE (:Concept {key: 'dup'})")
    with pytest.raises(Exception, match="(?i)constraint|already exists"):
        neo.run("CREATE (:Concept {key: 'dup'})")

def test_rewriting_one_reel_preserves_shared_concepts(neo):
    neo.write_reel(doc("A", [("cuda", "CUDA"), ("triton", "Triton")], [("triton", "DEPENDS_ON", "cuda")]))
    neo.write_reel(doc("B", [("cuda", "CUDA"), ("gpu", "GPU")]))
    neo.write_reel(doc("A", [("triton", "Triton")], h="h2"))
    assert keys(neo) == {"cuda", "triton", "gpu"}
    assert neo.run("MATCH (a:Assertion) RETURN count(a) AS n")[0]["n"] == 0
    assert neo.reel_hashes() == {"A": "h2", "B": "h1"}

def test_removing_a_reel_removes_only_what_it_owns(neo):
    neo.write_reel(doc("A", [("cuda", "CUDA"), ("triton", "Triton")], [("triton", "DEPENDS_ON", "cuda")]))
    neo.write_reel(doc("B", [("cuda", "CUDA"), ("gpu", "GPU")], [("gpu", "USES", "cuda")]))
    neo.remove_reel("A")
    assert keys(neo) == {"cuda", "gpu"}
    assert {r["id"] for r in neo.run("MATCH (p:Passage) RETURN p.chunk_id AS id")} == {"B:transcript:000"}
    assert [r["id"] for r in neo.run("MATCH (a:Assertion) RETURN a.id AS id")] == ["B-gpu-cuda"]

def test_failed_transaction_preserves_previous_graph(neo):
    neo.write_reel(doc("A", [("cuda", "CUDA"), ("triton", "Triton")], [("triton", "DEPENDS_ON", "cuda")]))
    before = neo.run("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c ORDER BY l")

    def write_then_fail(tx, d):
        st.write_reel_tx(tx, d)
        raise RuntimeError("crash inside the transaction")

    with pytest.raises(RuntimeError):
        neo._write(write_then_fail, doc("A", [("python", "Python")], h="h2"))
    assert neo.run("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c ORDER BY l") == before
    assert keys(neo) == {"cuda", "triton"} and neo.reel_hashes() == {"A": "h1"}

def test_offline_store_reports_graph_offline_and_closes_driver():
    s = st.Neo4jStore(uri=f"bolt://127.0.0.1:{free_port()}", user="neo4j", password="x" * 32)
    with pytest.raises(st.GraphOffline):
        s.ping()
    assert s._driver is None

def test_sync_end_to_end_and_review_survives_refresh(neo):
    triton_reel("A")
    triton_reel("B", extra="for GPU kernels")
    indexer.sync_all()
    llm = FakeLLM()
    r = gs.sync(store=neo, llm=llm)
    assert r["written"] == 2 and r["ready"] == 2
    assert q.verify(neo)["ok"], q.verify(neo)["problems"]
    rel = q.between(neo, "Triton", "CUDA")
    assert set(rel["relations"]) == {"corroborated"}
    assert {x["reel_id"] for x in rel["relations"]["corroborated"]} == {"A", "B"}
    assert rel["co_mentioned_in"] and rel["co_mention_note"].startswith("same Reel only")
    a = next(x for x in q.evidence(neo, "Triton", "CUDA") if x["reel_id"] == "A" and x["content_type"] == "transcript")
    assert a["quote"] == "Triton is built on top of CUDA" and a["source_url"].endswith("/reel/A/")
    gs.put_review("triton", "DEPENDS_ON", "cuda", a["chunk_id"], "accept", "checked by hand")
    gs.sync(store=neo, llm=llm)
    st_of = {x["assertion_id"]: x["status"] for x in q.evidence(neo, "Triton", "CUDA")}
    assert st_of[a["assertion_id"]] == "verified"
    triton_reel("A", extra="and more about Python")
    indexer.sync_all()
    gs.sync(store=neo, llm=llm)
    st_of = {x["assertion_id"]: x["status"] for x in q.evidence(neo, "Triton", "CUDA")}
    assert st_of[a["assertion_id"]] == "verified"
    gs.put_review("triton", "DEPENDS_ON", "cuda", a["chunk_id"], "reject", "actually wrong")
    gs.sync(store=neo, llm=llm)
    assert a["assertion_id"] not in {x["assertion_id"] for x in q.evidence(neo, "Triton", "CUDA")}
    assert a["assertion_id"] in {x["assertion_id"] for x in q.evidence(neo, "Triton", "CUDA", include_rejected=True)}
    assert q.verify(neo)["ok"]

def test_queries_and_phase4b_hooks(neo):
    triton_reel("A")
    add("B", "asyncio in Python with CUDA")
    indexer.sync_all()
    gs.sync(store=neo, llm=FakeLLM())
    shared = {x["key"]: x["n_reels"] for x in q.shared(neo)}
    assert shared == {"cuda": 2}
    rc = q.reel_concepts(neo, "A")
    assert {c["key"] for c in rc["concepts"]} >= {"triton", "cuda"} and rc["reel"]["source_url"].endswith("/reel/A/")
    cr = q.concept_reels(neo, "cuda")
    assert {r["reel_id"] for r in cr["reels"]} == {"A", "B"} and all(r["evidence"] for r in cr["reels"])
    chunk = cr["reels"][0]["evidence"][0]["chunk_id"]
    assert any(x["key"] == "cuda" for x in q.concepts_for_chunks(neo, [chunk]))
    nb = q.neighbors(neo, ["triton"])
    assert nb and nb[0]["to"] == "cuda" and nb[0]["status"] in ("proposed", "corroborated")

def test_verify_detects_orphans_stale_and_duplicates(neo):
    triton_reel("A")
    indexer.sync_all()
    gs.sync(store=neo, llm=FakeLLM())
    assert q.verify(neo)["ok"]
    neo.run("CREATE (:Concept {key: 'orphan', name: 'Orphan'})")
    neo.run("CREATE (:Concept {key: 'cudaq', name: 'CUDAQ'})-[:X]->(:Concept {key: 'cuda-q', name: 'CUDA-Q'})")
    neo.run("MATCH (r:Reel {reel_id: 'A'}) SET r.graph_hash = 'tampered'")
    v = q.verify(neo)
    text = "\n".join(v["problems"])
    assert not v["ok"]
    assert "concept without mentions or assertions" in text and "orphan" in text
    assert "graph hash differs" in text
    assert "near-duplicate" in text