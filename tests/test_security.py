"""Request guard: pinned extension ID, Host and Origin checks, and CORS for the extension only."""
import base64
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend import config, security
from backend.app import app

ROOT = Path(__file__).resolve().parent.parent

client = TestClient(app)
EXT = f"chrome-extension://{security.EXTENSION_ID}"

def test_pinned_id_matches_the_manifest_key():
    key = json.loads((ROOT / "extension" / "manifest.json").read_text())["key"]
    digest = hashlib.sha256(base64.b64decode(key)).hexdigest()[:32]
    assert "".join(chr(ord("a") + int(c, 16)) for c in digest) == security.EXTENSION_ID

def test_extension_and_local_pages_are_allowed():
    for origin in (EXT, "http://127.0.0.1:8765", "http://localhost:8765", "http://127.0.0.1:5173"):
        assert client.get("/api/health", headers={"Origin": origin}).status_code == 200, origin
    assert client.get("/api/health").status_code == 200

def test_other_extensions_and_websites_are_refused():
    for origin in ("chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "https://evil.example", "null",
                   "http://127.0.0.1.evil.example"):
        r = client.get("/api/health", headers={"Origin": origin})
        assert r.status_code == 403 and "origin" in r.text, origin
    r = client.post("/api/session/x/finish", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403

def test_dns_rebinding_host_is_refused_everywhere():
    for path in ("/api/health", "/workspace/", "/"):
        r = client.get(path, headers={"Host": "attacker.example:8765"})
        assert r.status_code == 403 and "host" in r.text, path

def test_cross_site_requests_without_origin_cannot_reach_the_api():
    img = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "image"}
    assert client.get("/api/session?source=test", headers=img).status_code == 403
    nav = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}
    assert client.get("/api/status", headers=nav).status_code == 403
    assert client.get("/workspace/", headers=nav).status_code != 403
    assert client.get("/api/health", headers={"Sec-Fetch-Site": "none", "Sec-Fetch-Mode": "cors"}).status_code == 200

def test_cors_answers_only_the_extension():
    pre = {"Access-Control-Request-Method": "POST"}
    ok = client.options("/api/process", headers={"Origin": EXT, **pre})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == EXT
    other = client.options("/api/process", headers={"Origin": "chrome-extension://bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", **pre})
    assert other.status_code == 403 and "access-control-allow-origin" not in other.headers

def test_extra_extension_ids_can_be_allowed(monkeypatch):
    monkeypatch.setattr(config, "EXTRA_EXTENSION_IDS", ("cccccccccccccccccccccccccccccccc",))
    assert security.origin_allowed("chrome-extension://cccccccccccccccccccccccccccccccc")
    assert not security.origin_allowed("chrome-extension://dddddddddddddddddddddddddddddddd")

def test_host_parsing():
    assert security.host_allowed("127.0.0.1:8765") and security.host_allowed("localhost") and security.host_allowed("[::1]:8765")
    assert not security.host_allowed(None) and not security.host_allowed("") and not security.host_allowed("evil.example")