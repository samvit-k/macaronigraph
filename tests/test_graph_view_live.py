"""Phase 5 read-only graph views (backend/graph/view.py) on a THROWAWAY Neo4j instance (fixture from
test_graph_neo4j_live; production output/neo4j is never touched). Skipped when Neo4j is not installed."""
import pytest

from backend.graph import view
from test_graph_neo4j_live import ROOT, clean, neo  # noqa: F401 — fixtures

pytestmark = pytest.mark.skipif(not (ROOT / ".tools" / "neo4j" / "bin" / "neo4j").exists(),
                                reason="Neo4j not installed (scripts/neo4j.sh install)")

SEED = """
CREATE (r1:Reel {reel_id:'R1', owner:'alice', caption_short:'CUDA basics', source_url:'https://www.instagram.com/reel/R1/', saved_at:'2026-11-02'})
CREATE (r2:Reel {reel_id:'R2', owner:'bob', caption_short:'GPU kernels', source_url:'https://www.instagram.com/reel/R2/', saved_at:'2026-11-01'})
CREATE (p1:Passage {chunk_id:'R1:transcript:000', content_type:'transcript', start:0.0, end:5.0, text:'CUDA runs kernels on the GPU.'})
CREATE (p2:Passage {chunk_id:'R2:ocr:001', content_type:'ocr', start:null, end:null, text:'CUDA and Triton'})
CREATE (cuda:Concept {key:'cuda', name:'CUDA', type:'technology', aliases:['CUDA']})
CREATE (tri:Concept {key:'triton', name:'Triton', type:'tool', aliases:['Triton']})
CREATE (gpu:Concept {key:'gpu', name:'GPU', type:'technology', aliases:['GPU']})
CREATE (r1)-[:HAS_PASSAGE]->(p1), (r2)-[:HAS_PASSAGE]->(p2)
CREATE (p1)-[:MENTIONS {surface:'CUDA'}]->(cuda), (p1)-[:MENTIONS {surface:'GPU'}]->(gpu)
CREATE (p2)-[:MENTIONS {surface:'CUDA'}]->(cuda), (p2)-[:MENTIONS {surface:'Triton'}]->(tri)
CREATE (a:Assertion {id:'a1', subject_key:'cuda', object_key:'gpu', predicate:'USES', status:'proposed', reel_id:'R1',
                     chunk_id:'R1:transcript:000', quote:'CUDA runs kernels on the GPU'})
CREATE (a)-[:SUBJECT]->(cuda), (a)-[:OBJECT]->(gpu), (a)-[:EVIDENCE]->(p1)
CREATE (x:Assertion {id:'a2', subject_key:'triton', object_key:'cuda', predicate:'DEPENDS_ON', status:'rejected', reel_id:'R2',
                     chunk_id:'R2:ocr:001', quote:'CUDA and Triton'})
CREATE (x)-[:SUBJECT]->(tri), (x)-[:OBJECT]->(cuda), (x)-[:EVIDENCE]->(p2)
"""

@pytest.fixture
def seeded(neo):
    neo.run(SEED)
    return neo

def test_overview_shows_reels_bridge_concepts_mentions_and_non_rejected_assertions(seeded):
    o = view.overview(seeded)
    assert o["totals"] == {**o["totals"], "reels": 2, "concepts": 3, "bridge": 1}
    assert [r["reel_id"] for r in o["reels"]] == ["R1", "R2"]                       # newest first
    keys = {c["key"]: c for c in o["concepts"]}
    assert keys["cuda"]["bridge"] and keys["cuda"]["n_reels"] == 2
    assert "gpu" in keys and not keys["gpu"]["bridge"]                               # endpoint of a shown assertion
    assert "triton" not in keys                                                      # single-Reel, only in a rejected one
    assert {(m["reel_id"], m["concept_key"]) for m in o["mentions"]} == {("R1", "cuda"), ("R2", "cuda"), ("R1", "gpu")}
    assert [a["id"] for a in o["assertions"]] == ["a1"] and o["assertions"][0]["status"] == "proposed"

def test_overview_pages_reels(seeded):
    o = view.overview(seeded, reel_offset=1, reel_limit=1)
    assert [r["reel_id"] for r in o["reels"]] == ["R2"] and o["totals"]["reels_shown"] == 1

def test_reel_neighborhood_has_all_its_concepts_with_excerpts(seeded):
    n = view.neighborhood(seeded, "reel", "R2")
    assert n["found"] and {c["key"] for c in n["concepts"]} == {"cuda", "triton"}
    ev = next(c for c in n["concepts"] if c["key"] == "triton")["evidence"][0]
    assert ev["content_type"] == "ocr" and ev["text"] == "CUDA and Triton"
    assert n["assertions"] == []                                                     # the rejected one is hidden

def test_concept_neighborhood_resolves_names_and_lists_reels(seeded):
    n = view.neighborhood(seeded, "concept", "cuda")
    assert n["concept"]["n_reels"] == 2 and {r["reel_id"] for r in n["reels"]} == {"R1", "R2"}
    assert [a["id"] for a in n["assertions"]] == ["a1"] and n["related_concepts"][0]["key"] == "gpu"
    assert view.neighborhood(seeded, "concept", "nothing-like-this")["found"] is False
    with pytest.raises(ValueError):
        view.neighborhood(seeded, "passage", "x")

def test_search_ranks_exact_and_prefix_matches_and_finds_reels(seeded):
    s = view.search(seeded, "cu")
    assert s["concepts"][0]["key"] == "cuda" and {r["reel_id"] for r in s["reels"]} == {"R1"}
    assert view.search(seeded, "   ") == {"query": "", "concepts": [], "reels": []}

def test_views_never_write(seeded):
    before = seeded.run("MATCH (n) RETURN count(n) AS n")[0]["n"]
    view.overview(seeded); view.neighborhood(seeded, "reel", "R1"); view.search(seeded, "gpu")
    assert seeded.run("MATCH (n) RETURN count(n) AS n")[0]["n"] == before