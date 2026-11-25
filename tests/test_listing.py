"""Deterministic listings over a real temporary Chroma index: detection, exact matching and scope."""

import pytest

from backend import answer, ask, config, conversations, embedder, indexer, jobs, listing, models
from test_indexer import add, oseg
from test_rag_retrieve import FakeGraph

class NoModel:
    def generate(self, *a, **k):
        raise AssertionError("the model must not be called for a deterministic listing")

@pytest.fixture
def idx(isolated_output, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    monkeypatch.setattr(ask, "_queue", __import__("collections").deque())
    monkeypatch.setattr(config, "RAG_HEAVY_WAIT_S", 0.5)
    monkeypatch.setitem(answer.DEFAULTS, "listing", True)
    jobs._export_state.update(pending=0, last=0.0)
    add("A", "CUDA kernels run on thousands of GPU cores and CUDA is fast")
    add("B", "Triton is built on top of CUDA for kernels")
    add("C", "sourdough bread starter recipe", ocr=[oseg(1, 2, "Made with CUDA on screen only")])
    add("D", "leet code problems for interviews")
    add("E", "a cudamalloc is not a mention")
    indexer.sync_all()
    yield
    indexer.release_writer()

@pytest.mark.parametrize("q,kind,term", [
    ("Which Reels mention CUDA?", "list", "CUDA"),
    ("Which of my saved Reels mention Claude Code?", "list", "Claude Code"),
    ("How many Reels mention Docker?", "count", "Docker"),
    ("How many of my Reels mention Python?", "count", "Python"),
    ("Which of them also mention Kubernetes?", "within", "Kubernetes"),
    ("Which other Reels mention this?", "list", None),
])
def test_detects_pure_listing_and_count_questions(q, kind, term):
    d = listing.detect(q)
    assert d and d["kind"] == kind and d["term"] == term

@pytest.mark.parametrize("q", [
    "Which Reels mention Docker, and in what context?", "Which Reels talk about CUDA?",
    "Which Reels recommend NeetCode?", "Which Reel teaches Python specifically for coding interviews?",
    "What is CUDA?", "Which Reels mention CUDA for beginners?",
])
def test_constrained_or_semantic_questions_are_not_deterministic(q):
    assert listing.detect(q) is None

def test_lists_every_literal_match_once_with_evidence_type_and_citations(idx):
    out, sources = listing.answer("Which Reels mention CUDA?", graph=FakeGraph())
    det = out["deterministic"]
    assert det["reels"] == ["A", "B", "C"] and det["ocr_only"] == ["C"] and "E" not in det["reels"]
    assert len(sources) == 3 and len({s["reel_id"] for s in sources}) == 3
    for s, c in zip(sources, out["citations"]):
        assert "cuda" in s["text"].lower() and c["chunk_id"] == s["chunk_id"] and c["sid"] == s["sid"]
    assert [x for x in out["sentences"] if x["citations"]][-1]["text"].count("on-screen text") == 1
    assert det["searched_reels"] == 5 and "5 processed Reels" in out["text"] and "not processed" in out["text"]

def test_counts_report_the_spoken_and_on_screen_split(idx):
    out, _ = listing.answer("How many Reels mention CUDA?", graph=FakeGraph())
    assert out["sentences"][0]["text"].startswith("3 of the 5 processed Reels mention")
    assert "2 in the spoken transcript or caption and 1 only in on-screen text" in out["sentences"][0]["text"]

def test_spacing_variants_match_and_zero_matches_are_stated(idx):
    out, _ = listing.answer("Which Reels mention LeetCode?", graph=FakeGraph())
    assert out["deterministic"]["reels"] == ["D"]
    out, sources = listing.answer("Which Reels mention Kubernetes?", graph=FakeGraph())
    assert not sources and out["sentences"][0]["text"].startswith("None of the 5 processed Reels mention")

def test_previous_reel_scoping(idx):
    out, _ = listing.answer("Which of them also mention Triton?", scope={"within": ["A", "B"]}, graph=FakeGraph())
    assert out["deterministic"]["reels"] == ["B"] and out["deterministic"]["scope"] == "within_previous"
    out, _ = listing.answer("Which other Reels mention this?", scope={"exclude": ["A"], "previous_term": "CUDA"},
                            graph=FakeGraph())
    assert out["deterministic"]["reels"] == ["B", "C"]
    assert listing.answer("Which of them mention CUDA?", scope={}, graph=FakeGraph()) is None

def test_stale_index_entries_are_skipped_and_reported(idx):
    with jobs.tx() as con:
        con.execute("UPDATE index_state SET content_hash='stale' WHERE reel_id='B'")
    out, _ = listing.answer("Which Reels mention CUDA?", graph=FakeGraph())
    assert "B" not in out["deterministic"]["reels"] and out["deterministic"]["skipped_stale"] == ["B"]
    assert not out["deterministic"]["complete"] and "out of date" in out["text"]

def test_graph_offline_still_answers_with_the_exact_wording(idx):
    class Down:
        def concept_index(self):
            raise ConnectionError("neo4j offline")
    out, _ = listing.answer("Which Reels mention CUDA?", graph=Down())
    assert out["deterministic"]["reels"] == ["A", "B", "C"] and "graph aliases unavailable" in out["text"]

def test_api_answers_listings_without_the_model_even_while_capture_is_active(idx, monkeypatch):
    monkeypatch.setattr(models, "shared_llm", lambda *a: NoModel())
    jobs.create_session("instagram-live", "instagram", 0)
    r = ask.ask("Which Reels mention CUDA?", graph=FakeGraph(), llm=NoModel())
    assert r["status"] == "ok" and {c["reel_id"] for c in r["citations"]} == {"A", "B", "C"}
    assert all(any(s["chunk_id"] == c["chunk_id"] for s in r["sources"]) for c in r["citations"])
    t = conversations.get_turn(r["turn_id"])
    assert t["cited_reels"] == ["A", "B", "C"] and t["retrieval"]["intent"] == "listing_list"
    r2 = ask.ask("Which of them also mention Triton?", conversation_id=r["conversation_id"], graph=FakeGraph(),
                 llm=NoModel())
    assert r2["status"] == "ok" and [c["reel_id"] for c in r2["citations"]] == ["B"]

def test_listing_is_off_unless_enabled(idx, monkeypatch):
    monkeypatch.setitem(answer.DEFAULTS, "listing", False)
    assert answer.deterministic("Which Reels mention CUDA?", {"sources": []}) is None