"""Phase 4D passage selection within already-retrieved Reels (backend/evidence.py) over a real temporary index:
answer-bearing passages of a pack Reel are added; no new Reels; nothing removed or reordered; caps and stale
content hashes respected."""
import pytest

from backend import config, embedder, evidence, indexer, jobs
from test_chunking import make_record, seg

FILLER = " ".join(["The creator talks about many general things in this long part of the video."] * 5)
ROUNDS = [FILLER + " Round 1 is a paid two-day take-home with an open brief.",
          FILLER + " Round 2 is hands-on service design with queues and caches.",
          FILLER + " Round 3 covers team values and communication."]


def add_reel(rid, texts):
    segs = [seg(i * 20, i * 20 + 19, t) for i, t in enumerate(texts)]
    rec = make_record(segs, [], rid=rid, caption=f"About {rid}")
    rec.update(session_id=f"instagram-{rid}", processed_at="2026-11-23T00:00:01+00:00", errors=[], warnings=[])
    return jobs.record_attempt(rec, export=False)


@pytest.fixture
def idx(isolated_output, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    jobs._export_state.update(pending=0, last=0.0)
    add_reel("H", ROUNDS)
    add_reel("Z", ["Completely unrelated cooking video about bread. " * 8])
    indexer.sync_all()
    yield indexer.get_collection(indexer.active_collection())
    indexer.release_writer()


def pack_with(col, chunk_id):
    got = col.get(ids=[chunk_id], include=["documents", "metadatas"])
    md = got["metadatas"][0]
    return {"question": "q", "sources": [{"sid": "S1", "chunk_id": chunk_id, "reel_id": md["reel_id"],
                                          "text": got["documents"][0], "content_type": md["content_type"],
                                          "similarity": 0.5}]}


def test_adds_the_answer_bearing_passage_of_a_reel_already_in_the_pack(idx):
    ids = sorted(i for i in idx.get(where={"reel_id": "H"})["ids"] if ":transcript:" in i)
    assert len(ids) >= 3
    first = next(i for i in ids if "Round 1" in idx.get(ids=[i])["documents"][0])
    r = evidence.select("What happens in round 2?", pack_with(idx, first), {"sim_floor": 0.0})
    assert r["sources"][0]["chunk_id"] == first                                  # existing source untouched
    added = r["sources"][1:]
    assert added and "Round 2" in added[0]["text"] and added[0]["sid"] == "S2"
    assert {s["reel_id"] for s in r["sources"]} == {"H"}                          # no new Reels
    assert added[0]["channels"][0]["type"] == "same_reel_passage"


def test_caps_and_no_lexical_match_means_nothing_added(idx):
    ids = sorted(i for i in idx.get(where={"reel_id": "H"})["ids"] if ":transcript:" in i)
    r = evidence.select("Quantum chromodynamics lattice?", pack_with(idx, ids[0]), {"sim_floor": 0.0})
    assert len(r["sources"]) == 1 and r["passage_select"]["added"] == []
    r = evidence.select("round", pack_with(idx, ids[0]), {"sim_floor": 0.0, "max_add": 1})
    assert len(r["passage_select"]["added"]) <= 1


def test_stale_reels_are_not_used(idx):
    ids = sorted(i for i in idx.get(where={"reel_id": "H"})["ids"] if ":transcript:" in i)
    with jobs.tx() as con:
        con.execute("UPDATE index_state SET content_hash='stale' WHERE reel_id='H'")
    r = evidence.select("What happens in round 2?", pack_with(idx, ids[0]), {"sim_floor": 0.0})
    assert r["passage_select"]["added"] == []


def test_empty_pack_is_returned_unchanged():
    r = evidence.select("anything", {"sources": []})
    assert r["sources"] == [] and r["passage_select"]["added"] == []
