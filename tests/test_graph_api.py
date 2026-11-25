"""Graph API with Neo4j unavailable: graph endpoints report 503 and the rest of the backend is unaffected."""

import pytest
from fastapi.testclient import TestClient

from backend import config, embedder, indexer, jobs
from backend.app import app
from helpers import free_port, production_path_guard
from test_indexer import add

client = TestClient(app)

@pytest.fixture(autouse=True)
def offline(isolated_output, monkeypatch):
    monkeypatch.setattr(config, "NEO4J_URI", f"bolt://127.0.0.1:{free_port()}")
    monkeypatch.setattr(config, "NEO4J_PASSWORD", "x" * 32)
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    jobs._export_state.update(pending=0, last=0.0)
    with production_path_guard(isolated_output):
        yield

@pytest.mark.parametrize("path", ["/api/graph/verify", "/api/graph/reel/A", "/api/graph/concept?name=CUDA",
                                  "/api/graph/shared", "/api/graph/between?a=CUDA&b=Triton",
                                  "/api/graph/evidence?a=CUDA&b=Triton"])
def test_graph_queries_report_503_when_neo4j_is_offline(path):
    r = client.get(path)
    assert r.status_code == 503 and "offline" in r.json()["detail"]

def test_status_reports_offline_and_backend_keeps_working(monkeypatch):
    add("A", "Triton is built on top of CUDA")
    indexer.sync_all()
    st = client.get("/api/graph/status").json()
    assert st["neo4j"]["online"] is False and st["eligible"] == 1
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/search", params={"q": "Triton CUDA"}).json()["results"][0]["reel_id"] == "A"

def test_manual_sync_while_offline_does_not_fail_and_needs_no_llm_when_deferred(monkeypatch):
    add("A", "Triton is built on top of CUDA")
    jobs.create_session("instagram-live", "instagram", 0)
    from backend.graph import sync as gs
    monkeypatch.setattr(gs, "extractor_version", lambda model_id=None: "x-test")
    r = client.post("/api/graph/sync", params={"wait": 1}).json()
    assert r["online"] is False and r["deferred"] == 1 and r["extracted"] == 0

def test_unknown_request_id_is_404():
    assert client.get("/api/graph/request/999999").status_code == 404