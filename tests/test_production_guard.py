"""The test-session production guard (conftest.py): tests can never reach the real output directory (SQLite
pipeline.db, Chroma), nor the production Neo4j — even if a test forgets isolated_output or restores patches."""
import sqlite3

import chromadb
import pytest

import conftest
from backend import config, indexer, jobs
from backend.graph.store import Neo4jStore
from conftest import PRODUCTION_NEO4J, ROOT, SESSION_DIR, ProductionAccess, _under_production

def test_defaults_point_at_the_session_directory_not_production():
    for p in (config.OUTPUT_DIR, jobs.db_path(), config.chroma_dir(), config.OUTPUT_JSONL, config.TMP_ROOT):
        assert SESSION_DIR in p.parents and not _under_production(p), p

def test_restoring_patches_returns_to_the_session_directory(tmp_path):
    """What leaked before: a restored monkeypatch put OUTPUT_DIR back to the real output/. Now the restored
    value is the session directory."""
    before = config.OUTPUT_DIR
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    assert config.OUTPUT_DIR == before and not _under_production(config.OUTPUT_DIR)

@pytest.mark.parametrize("target", ["output/pipeline.db", "output/chroma/chroma.sqlite3"])
def test_sqlite_refuses_production_files_before_any_statement(target):
    path = ROOT / target
    with pytest.raises(ProductionAccess):
        sqlite3.connect(path)
    with pytest.raises(ProductionAccess):
        sqlite3.connect(f"file:{path}?mode=ro", uri=True)

def test_pipeline_db_connect_is_refused_when_pointed_at_production(monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", ROOT / "output")
    with pytest.raises(ProductionAccess):
        with jobs.connect():
            pass

def test_isolated_databases_still_work(isolated_output):
    with jobs.connect() as con:
        assert con.execute("SELECT 1").fetchone()[0] == 1
    assert isolated_output in jobs.db_path().parents

def test_neo4j_production_uri_and_password_are_refused():
    assert config.NEO4J_PASSWORD is None and config.NEO4J_URI != PRODUCTION_NEO4J["uri"]
    with pytest.raises(ProductionAccess):
        Neo4jStore(uri=PRODUCTION_NEO4J["uri"], password="x" * 32).ping()
    if PRODUCTION_NEO4J["password"]:
        with pytest.raises(ProductionAccess):
            Neo4jStore(uri="bolt://127.0.0.1:9", password=PRODUCTION_NEO4J["password"]).ping()

def _snapshot(d):
    """(relative path, size, mtime_ns) of every file — via stat only, no file is opened."""
    return sorted((str(p.relative_to(d)), p.stat().st_size, p.stat().st_mtime_ns) for p in d.rglob("*") if p.is_file())

def test_temporary_chroma_client_works(isolated_output, monkeypatch):
    monkeypatch.setattr(indexer, "_clients", {})
    c = indexer.client()
    col = c.get_or_create_collection("guard_probe")
    col.add(ids=["a"], embeddings=[[0.1, 0.2, 0.3]], documents=["probe"])
    assert col.get(ids=["a"])["documents"] == ["probe"]
    assert list(indexer._clients) == [str(config.chroma_dir())]
    assert isolated_output in config.chroma_dir().parents
    assert (config.chroma_dir() / "chroma.sqlite3").exists()                       # really written, in tmp

def test_production_chroma_client_is_blocked_before_any_file_is_touched(monkeypatch):
    prod = ROOT / "output" / "chroma"
    before = _snapshot(prod) if prod.exists() else None
    calls = []
    monkeypatch.setattr(conftest, "_real_persistent_client", lambda *a, **k: calls.append(a) or pytest.fail(
        "the real Chroma client was constructed for a production path"))
    with pytest.raises(ProductionAccess):
        chromadb.PersistentClient(path=str(prod))
    monkeypatch.setattr(config, "OUTPUT_DIR", ROOT / "output")                     # via the application path too
    monkeypatch.setattr(indexer, "_clients", {})
    with pytest.raises(ProductionAccess):
        indexer.client()
    assert calls == [] and indexer._clients == {}
    assert (_snapshot(prod) if prod.exists() else None) == before                  # nothing created or modified