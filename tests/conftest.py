"""Test-session isolation: a temporary OUTPUT_DIR per test and a guard that refuses production data."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# FastAPI's TestClient sends Host: testserver; the request guard must accept it in tests only.
os.environ.setdefault("ALLOWED_HOSTS", "127.0.0.1,localhost,::1,testserver")

from backend import config  # noqa: E402

PLAY_WALL = 1_800_000_000_000

PRODUCTION_OUTPUT = {(ROOT / "output").resolve(), Path(config.OUTPUT_DIR).resolve()}
PRODUCTION_NEO4J = {"uri": config.NEO4J_URI, "password": config.NEO4J_PASSWORD}
SESSION_DIR = Path(tempfile.mkdtemp(prefix="reel-tests-"))

class ProductionAccess(AssertionError):
    pass

def _under_production(path) -> bool:
    s = str(path)
    if s.startswith("file:"):
        s = s[5:].split("?", 1)[0]
    if not s or s == ":memory:":
        return False
    p = Path(s).resolve()
    return any(p == d or d in p.parents for d in PRODUCTION_OUTPUT)

_real_sqlite_connect = sqlite3.connect

def _guarded_sqlite_connect(database, *args, **kwargs):
    if _under_production(database):
        raise ProductionAccess(f"test tried to open a production database: {database}")
    return _real_sqlite_connect(database, *args, **kwargs)

sqlite3.connect = _guarded_sqlite_connect
config.OUTPUT_DIR = SESSION_DIR / "out"
config.OUTPUT_JSONL = SESSION_DIR / "out" / "enriched_reels.jsonl"
config.TMP_ROOT = SESSION_DIR / "tmp"
config.NEO4J_URI = "bolt://127.0.0.1:9"
config.NEO4J_PASSWORD = None

from backend.graph import store as _graph_store  # noqa: E402

_real_driver = _graph_store.Neo4jStore.driver

def _guarded_driver(self):
    if self.uri == PRODUCTION_NEO4J["uri"] or (PRODUCTION_NEO4J["password"]
                                               and self.password == PRODUCTION_NEO4J["password"]):
        raise ProductionAccess(f"test tried to connect to the production Neo4j ({self.uri})")
    return _real_driver(self)

_graph_store.Neo4jStore.driver = _guarded_driver

import chromadb  # noqa: E402

_real_persistent_client = chromadb.PersistentClient

def _guarded_persistent_client(path=None, *args, **kwargs):
    if path is None or _under_production(path):
        raise ProductionAccess(f"test tried to open a production Chroma directory: {path}")
    return _real_persistent_client(path, *args, **kwargs)

chromadb.PersistentClient = _guarded_persistent_client

_real_mkdir = Path.mkdir

def _guarded_mkdir(self, *args, **kwargs):
    if _under_production(self):
        raise ProductionAccess(f"test tried to create a production directory: {self}")
    return _real_mkdir(self, *args, **kwargs)

Path.mkdir = _guarded_mkdir

def ground_truth() -> dict:
    return json.loads(config.TEST_GROUND_TRUTH.read_text())

def clip_upload(rid: str, every_s: float = 1.0) -> tuple[bytes | None, list[bytes], list[int]]:
    path = config.TEST_MEDIA_DIR / f"{rid}.mp4"
    audio = subprocess.run([config.FFMPEG, "-v", "error", "-i", str(path), "-vn", "-c:a", "libopus",
                            "-f", "webm", "-"], capture_output=True, check=True).stdout
    raw = subprocess.run([config.FFMPEG, "-v", "error", "-i", str(path), "-vf", f"fps={1 / every_s}",
                          "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"],
                         capture_output=True, check=True).stdout
    frames = [b"\xff\xd8" + p for p in raw.split(b"\xff\xd8")[1:]]
    walls = [PLAY_WALL + int((i * every_s + every_s / 2) * 1000) for i in range(len(frames))]
    return audio, frames, walls

@pytest.fixture(autouse=True)
def _no_real_local_llm(monkeypatch):
    from backend import models
    from backend.graph import llm as graph_llm

    class _Refuse:
        def __init__(self, *a, **k):
            raise AssertionError("a test tried to load the real local LLM; pass or patch a fake")
    monkeypatch.setattr(graph_llm, "MlxLLM", _Refuse)
    monkeypatch.setattr(models, "_shared", None)
    monkeypatch.setattr(models, "_llms", {})

@pytest.fixture(autouse=True)
def _sync_processing(monkeypatch):
    monkeypatch.setattr(config, "ASYNC_PROCESSING", False)

@pytest.fixture
def isolated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "OUTPUT_JSONL", tmp_path / "out" / "enriched_reels.jsonl")
    monkeypatch.setattr(config, "TMP_ROOT", tmp_path / "tmp")
    return tmp_path