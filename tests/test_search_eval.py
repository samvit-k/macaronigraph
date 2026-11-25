"""Retrieval-eval scoring logic (no model): query validation and OCR passage-level hits."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import search_eval as ev  # noqa: E402

def chunk(cid, rid, kind, text, quality="normal"):
    return {"id": cid, "text": text, "metadata": {"reel_id": rid, "content_type": kind, "ocr_quality": quality}}

def test_ocr_only_queries_without_valid_terms_are_excluded():
    qs = [{"q": "a", "relevant": ["R1"], "kind": "ocr_only"},                       # no terms
          {"q": "b", "relevant": ["R1"], "kind": "ocr_only", "ocr_terms": []},      # empty → all([]) trap
          {"q": "c", "relevant": ["R1"], "kind": "ocr_only", "ocr_terms": ["", " "]},
          {"q": "d", "relevant": ["R1"], "kind": "ocr_only", "ocr_terms": ["Nav2"]},
          {"q": "e", "relevant": ["R1"], "kind": "semantic"}]
    scored, excluded, warnings = ev.validate(qs, {"R1"})
    assert [q["q"] for q in scored] == ["d", "e"]
    assert [q["q"] for q in excluded] == ["a", "b", "c"] and all("ocr_terms" in q["reason"] for q in excluded)
    assert not ev.contains_terms("anything", [])                                   # never a vacuous hit

def test_missing_relevant_ids_are_reported_and_fully_missing_queries_excluded():
    qs = [{"q": "a", "relevant": ["GONE", "R1"], "kind": "semantic"},
          {"q": "b", "relevant": ["GONE"], "kind": "semantic"}]
    scored, excluded, warnings = ev.validate(qs, {"R1"})
    assert scored[0]["relevant"] == ["R1"] and scored[0]["missing"] == ["GONE"]
    assert [q["q"] for q in excluded] == ["b"] and len(warnings) == 2

def test_ocr_hit_found_in_a_lower_scoring_chunk_of_the_same_reel():
    chunks = [chunk("R1:ocr:000", "R1", "ocr", "robotics intro slide"),            # best OCR chunk, wrong text
              chunk("R1:transcript:000", "R1", "transcript", "Nav2 SLAM spoken"),  # speech must not count
              chunk("R2:ocr:000", "R2", "ocr", "unrelated"),
              chunk("R1:ocr:001", "R1", "ocr", "Nav2 + autonomous SLAM mapping")]  # the answer
    sims = [0.9, 0.95, 0.8, 0.7]
    ranked = ev.rank_ocr_passages(sims, chunks)
    assert [c["id"] for _, c in ranked] == ["R1:ocr:000", "R2:ocr:000", "R1:ocr:001"]   # transcript excluded
    rank, c = ev.ocr_passage_rank(ranked, ["R1"], ["nav2", "slam"])
    assert rank == 3 and c["id"] == "R1:ocr:001"
    assert ev.ocr_passage_rank(ranked, ["R1"], ["gazebo"]) == (None, None)
    # Reel level still groups by Reel and may count the transcript
    assert ev.reel_rank(ev.rank_reels(sims, chunks), ["R1"]) == 1

def test_fragment_weight_applies_at_passage_level():
    chunks = [chunk("A", "R1", "ocr_fragment", "Nav2 SLAM", quality="fragment"), chunk("B", "R2", "ocr", "x")]
    ranked = ev.rank_ocr_passages([0.9, 0.8], chunks)
    assert ranked[0][1]["id"] == "B"                                               # 0.9 × 0.85 < 0.8

def test_fixture_ocr_only_queries_all_have_terms():
    path = ROOT / "fixtures" / "search_queries.json"
    if not path.exists():                       # built from the owner's saved Reels; not part of the public repository
        pytest.skip("private search query set not present")
    qs = json.loads(path.read_text())["queries"]
    ocr = [q for q in qs if q["kind"] == "ocr_only"]
    assert ocr and all(ev._valid_terms(q.get("ocr_terms")) for q in ocr)