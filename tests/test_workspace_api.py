"""Phase 5 additive backend surface: welcome page, extension download (explicit allowlist), workspace mount,
graph-view routes when Neo4j is offline, and the additive `cited_concepts` field. The legacy /app/ page keeps working."""
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from backend import app as app_mod
from backend import ask
from backend.graph import store as st

client = TestClient(app_mod.app)


def test_extension_zip_contains_exactly_the_allowlist():
    r = client.get("/welcome/extension.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "macaronigraph-extension.zip" in r.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names == [f"macaronigraph-extension/{f}" for f in app_mod.EXTENSION_FILES]
    for n in names:                                       # nothing secret, local or developmental can be included
        base = n.rsplit("/", 1)[-1]
        assert not base.startswith(".") and base not in ("manifest.previous.json", "manifest.next.json")
        assert not base.endswith((".env", ".local", ".key", ".pem", ".db", ".jsonl", ".log"))
    manifest = zipfile.ZipFile(io.BytesIO(r.content)).read("macaronigraph-extension/manifest.json").decode()
    assert '"name": "MacaroniGraph"' in manifest


def test_allowlist_covers_every_file_the_manifest_and_popup_reference():
    import json
    import re
    root = app_mod.EXTENSION_DIR
    m = json.loads((root / "manifest.json").read_text())
    refs = {m["background"]["service_worker"], m["action"]["default_popup"], *m["icons"].values()}
    popup = (root / m["action"]["default_popup"]).read_text()
    refs |= {f"ui-next/{x}" for x in re.findall(r'(?:src|href)="([^":]+)"', popup)}
    bg = (root / "background.js").read_text()
    refs |= set(re.findall(r'["\']([a-z]+\.(?:js|html))["\']', bg))
    assert refs <= set(app_mod.EXTENSION_FILES), refs - set(app_mod.EXTENSION_FILES)


def test_welcome_and_workspace_are_served_when_built():
    if not (app_mod.WORKSPACE_DIR / "index.html").is_file():
        r = client.get("/welcome")
        assert r.status_code == 503 and "npm run build" in r.json()["detail"]
        pytest.skip("frontend not built")
    r = client.get("/welcome")
    assert r.status_code == 200 and "<title>MacaroniGraph" in r.text
    r = client.get("/workspace/")
    assert r.status_code == 200 and "<title>MacaroniGraph</title>" in r.text
    assert client.get("/workspace", follow_redirects=False).status_code in (307, 302)


def test_legacy_app_redirects_and_worker_tab_still_works():
    assert client.get("/app/", follow_redirects=False).headers["location"] == "/workspace/"
    r = client.get("/")
    assert r.status_code == 200 and "worker tab" in r.text


def test_graph_views_report_offline_as_503(monkeypatch):
    def offline(self, *a, **k):
        raise st.GraphOffline("Neo4j unavailable (test)")
    monkeypatch.setattr(st.Neo4jStore, "run", offline)
    for url in ("/api/graph/overview", "/api/graph/neighborhood?kind=reel&id=R1", "/api/graph/search?q=cuda"):
        r = client.get(url)
        assert r.status_code == 503 and "offline" in r.json()["detail"], url


def test_neighborhood_validates_kind():
    assert client.get("/api/graph/neighborhood?kind=passage&id=x").status_code == 422


def test_response_carries_cited_concepts_additively():
    t = {"id": 3, "conversation_id": "c", "question": "q", "status": "ok", "retrieval": {"sources": []},
         "answer": {"status": "ok", "text": "t", "sentences": [], "citations": []},
         "cited_concepts": [{"key": "cuda", "name": "CUDA", "type": "technology"}]}
    r = ask.response(t)
    assert r["cited_concepts"] == [{"key": "cuda", "name": "CUDA"}]
    assert {"citations", "sources", "answer", "status", "turn_id"} <= set(r)       # existing contract unchanged


def test_clear_all_conversations_endpoint(isolated_output):
    from backend import conversations as cv
    a = cv.create()
    cv.create()
    r = client.delete("/api/conversations")
    assert r.status_code == 200 and r.json() == {"deleted": 2}
    assert client.get("/api/conversations").json() == [] and cv.get(a) is None
