"""Phase 3 HTTP API: /api/search, /api/index/status, /api/index/sync, /api/health index summary."""
import pytest
from fastapi.testclient import TestClient

from backend import config, embedder, indexer, jobs
from backend.app import app
from test_chunking import oseg
from test_indexer import add

client = TestClient(app)   # no lifespan: the background worker/indexer threads are not started

@pytest.fixture(autouse=True)
def _iso(isolated_output, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    jobs._export_state.update(pending=0, last=0.0)
    yield
    indexer.release_writer()

def test_search_is_503_before_an_index_exists():
    r = client.get("/api/search", params={"q": "anything"})
    assert r.status_code == 503 and "no index" in r.json()["detail"]

def test_sync_then_grouped_search_shape():
    add("A", "sourdough bread baking", ocr=[oseg(1, 2, "Flour 500g\nWater 350g")])
    add("B", "python asyncio event loop")
    r = client.post("/api/index/sync", params={"wait": 1})
    assert r.status_code == 200 and r.json()["build"]["state"] == "activated"
    r = client.get("/api/search", params={"q": "asyncio event loop", "k": 2})
    body = r.json()
    assert r.status_code == 200 and body["group"] == "reel" and body["model"] == embedder.HashEmbedder.MODEL
    top = body["results"][0]
    assert top["reel_id"] == "B" and top["source_url"].endswith("/reel/B/") and top["owner_username"] == "someone"
    p = top["passages"][0]
    assert {"chunk_id", "content_type", "text", "start", "end", "score", "similarity", "passage_ids"} <= set(p)
    assert len(top["passages"]) <= 3

def test_raw_passages_and_type_filter():
    add("A", "sourdough", ocr=[oseg(1, 2, "Flour water salt")])
    client.post("/api/index/sync", params={"wait": 1})
    r = client.get("/api/search", params={"q": "flour", "group": "none", "types": "ocr"})
    hits = r.json()["results"]
    assert hits and all(h["content_type"].startswith("ocr") for h in hits) and all("reel_id" in h for h in hits)
    assert client.get("/api/search", params={"q": "x", "types": "bogus"}).status_code == 400
    assert client.get("/api/search", params={"q": "x", "group": "bogus"}).status_code == 400
    assert client.get("/api/search", params={"q": ""}).status_code == 422

def test_index_status_and_health_summary():
    add("A")
    add("F", "nope", status="failed")
    st = client.get("/api/index/status").json()
    assert st["active"] is None and st["pending"] == 1
    client.post("/api/index/sync", params={"wait": 1})
    st = client.get("/api/index/status").json()
    assert (st["eligible"], st["indexed"], st["pending"], st["errors"]) == (1, 1, 0, 0) and st["chunks"] > 0
    h = client.get("/api/health").json()["index"]
    assert h["active"] == st["active"] and h["build"] == "activated"

def test_sync_conflict_when_another_process_owns_the_index(monkeypatch):
    monkeypatch.setattr(indexer, "acquire_writer", lambda: False)
    assert client.post("/api/index/sync", params={"wait": 1}).status_code == 409