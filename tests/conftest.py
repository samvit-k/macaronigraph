import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402

PLAY_WALL = 1_800_000_000_000  # arbitrary epoch ms

# ---------------------------------------------------------------- production guard (installed before any test runs)
# 1. The defaults themselves point at a throwaway session directory, so code that forgets isolated_output — or a
#    monkeypatch.undo() — lands there, never in the real output/ directory.
# 2. sqlite3.connect refuses any database file under the real output directory (covers pipeline.db and Chroma's
#    chroma.sqlite3) — before a single statement can run.
# 3. Neo4j: the production URI/password from .env.local are removed from config, and any attempt to open a
#    driver to the production URI or with the production password is refused.
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
config.NEO4J_URI = "bolt://127.0.0.1:9"          # discard port: nothing listens
config.NEO4J_PASSWORD = None

from backend.graph import store as _graph_store  # noqa: E402

_real_driver = _graph_store.Neo4jStore.driver


def _guarded_driver(self):
    if self.uri == PRODUCTION_NEO4J["uri"] or (PRODUCTION_NEO4J["password"]
                                               and self.password == PRODUCTION_NEO4J["password"]):
        raise ProductionAccess(f"test tried to connect to the production Neo4j ({self.uri})")
    return _real_driver(self)


_graph_store.Neo4jStore.driver = _guarded_driver

# 4. Chroma: a client for a path under the real output directory is refused before any Chroma code runs (so no
#    file there is opened, created or migrated).
import chromadb  # noqa: E402

_real_persistent_client = chromadb.PersistentClient


def _guarded_persistent_client(path=None, *args, **kwargs):
    if path is None or _under_production(path):
        raise ProductionAccess(f"test tried to open a production Chroma directory: {path}")
    return _real_persistent_client(path, *args, **kwargs)


chromadb.PersistentClient = _guarded_persistent_client

# Creating a directory under the production output (e.g. an empty output/chroma in a fresh clone) is refused too,
# so a test can never leave anything behind in the real data folder.
_real_mkdir = Path.mkdir


def _guarded_mkdir(self, *args, **kwargs):
    if _under_production(self):
        raise ProductionAccess(f"test tried to create a production directory: {self}")
    return _real_mkdir(self, *args, **kwargs)


Path.mkdir = _guarded_mkdir


def ground_truth() -> dict:
    return json.loads(config.TEST_GROUND_TRUTH.read_text())


def clip_upload(rid: str, every_s: float = 1.0) -> tuple[bytes | None, list[bytes], list[int]]:
    """Simulate what the extension uploads for a test clip: webm/opus audio + 1 fps JPEG frames."""
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
    """Tests never load the real local LLM (slow, memory-heavy): constructing it fails unless a test patches
    graph.llm.MlxLLM with a fake. The shared model instance is reset around every test."""
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
    """Tests process uploads inline (same worker code path); async behaviour has dedicated tests."""
    monkeypatch.setattr(config, "ASYNC_PROCESSING", False)


@pytest.fixture
def isolated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "OUTPUT_JSONL", tmp_path / "out" / "enriched_reels.jsonl")
    monkeypatch.setattr(config, "TMP_ROOT", tmp_path / "tmp")
    return tmp_path
