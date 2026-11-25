"""scripts/data_digest.py — read-only, fail-closed integrity digest. Temporary fixtures only."""
import json
import sqlite3
import sys

import pytest

from backend import config

sys.path.insert(0, str(config.ROOT / "scripts"))
import data_digest as dd  # noqa: E402

NODES = [{"eid": "1", "labels": ["Reel"], "props": {"reel_id": "R1", "content_hash": "h"}},
         {"eid": "2", "labels": ["Passage"], "props": {"chunk_id": "R1:t:0", "text": "CUDA kernels"}},
         {"eid": "3", "labels": ["Concept"], "props": {"key": "cuda", "name": "CUDA"}},
         {"eid": "4", "labels": ["Concept"], "props": {"key": "triton", "name": "Triton"}}]
RELS = [{"type": "HAS_PASSAGE", "a": "1", "b": "2", "props": {}},
        {"type": "MENTIONS", "a": "2", "b": "3", "props": {"surface": "CUDA"}}]


def fake_query(nodes=NODES, rels=RELS):
    return lambda q: [dict(r) for r in (nodes if q == dd.NODES_Q else rels)]


@pytest.fixture
def prod(tmp_path):
    out = tmp_path / "out"
    (out / "chroma" / "seg-1").mkdir(parents=True)
    con = sqlite3.connect(out / "pipeline.db")
    con.executescript("CREATE TABLE reels(reel_id TEXT, status TEXT); CREATE TABLE attempts(id INT, reel_id TEXT);"
                      "CREATE TABLE sessions(id INT, finished_at INT, last_seen INT, created_at INT);"
                      "INSERT INTO reels VALUES('R1','ok'); INSERT INTO attempts VALUES(1,'R1');")
    con.commit(); con.close()
    con = sqlite3.connect(out / "chroma" / "chroma.sqlite3")
    con.executescript("CREATE TABLE embeddings_queue(id TEXT, vector BLOB); INSERT INTO embeddings_queue VALUES('c1', x'00ff');")
    con.commit(); con.close()
    for f in ("data_level0.bin", "header.bin", "length.bin", "link_lists.bin", "index_metadata.pickle"):
        (out / "chroma" / "seg-1" / f).write_bytes(b"\x01\x02" + f.encode())
    return out


def take(out, query=None):
    return dd.digest(out, query=query or fake_query())


def test_identical_digests(prod):
    a, b = take(prod), take(prod)
    assert a["complete"] and dd.compare(a, b) == ("IDENTICAL", [])
    assert set(a["chroma"]["segments"]["seg-1"]) == {"data_level0.bin", "header.bin", "length.bin",
                                                       "link_lists.bin", "index_metadata.pickle"}


def test_changed_canonical_record_is_detected(prod):
    a = take(prod)
    con = sqlite3.connect(prod / "pipeline.db"); con.execute("UPDATE reels SET status='partial'"); con.commit(); con.close()
    verdict, diff = dd.compare(a, take(prod))
    assert verdict == "DIFFERENT" and any(d.startswith("canonical.reels.sha256") for d in diff)


def test_changed_chroma_vector_and_modified_hnsw_file_are_detected(prod):
    a = take(prod)
    con = sqlite3.connect(prod / "chroma" / "chroma.sqlite3")
    con.execute("UPDATE embeddings_queue SET vector=x'0000'"); con.commit(); con.close()
    verdict, diff = dd.compare(a, take(prod))
    assert verdict == "DIFFERENT" and any("embeddings_queue" in d for d in diff)
    b = take(prod)
    (prod / "chroma" / "seg-1" / "link_lists.bin").write_bytes(b"tampered")
    verdict, diff = dd.compare(b, take(prod))
    assert verdict == "DIFFERENT" and any("segments.seg-1.link_lists.bin" in d for d in diff)


def test_changed_relationship_endpoint_or_property_is_detected(prod):
    a = take(prod)
    moved = [RELS[0], {**RELS[1], "b": "4"}]                     # MENTIONS now points at Triton instead of CUDA
    verdict, diff = dd.compare(a, take(prod, fake_query(rels=moved)))
    assert verdict == "DIFFERENT" and any("relationships_sha256" in d for d in diff)
    prop = [RELS[0], {**RELS[1], "props": {"surface": "Cuda"}}]
    assert dd.compare(a, take(prod, fake_query(rels=prop)))[0] == "DIFFERENT"


def test_endpoint_identity_ignores_internal_element_ids(prod):
    renumbered = [{**n, "eid": "x" + n["eid"]} for n in NODES]
    rels = [{**r, "a": "x" + r["a"], "b": "x" + r["b"]} for r in RELS]
    assert dd.compare(take(prod), take(prod, fake_query(renumbered, rels)))[0] == "IDENTICAL"


@pytest.mark.parametrize("nodes,why", [
    (NODES + [{"eid": "9", "labels": ["Mystery"], "props": {}}], "0 canonical ids"),
    (NODES + [{"eid": "9", "labels": ["Concept"], "props": {"key": "cuda"}}], "duplicate canonical id"),
])
def test_unidentifiable_or_ambiguous_nodes_fail_closed(prod, nodes, why):
    d = take(prod, fake_query(nodes=nodes))
    assert not d["complete"] and why in d["errors"]["neo4j"]


def test_unavailable_sources_make_the_digest_incomplete(prod):
    (prod / "chroma" / "chroma.sqlite3").unlink()
    d = dd.digest(prod)                                           # also: no Neo4j password in tests
    assert not d["complete"] and set(d["errors"]) == {"chroma", "neo4j"}


def test_matching_error_responses_are_never_identical(prod):
    (prod / "pipeline.db").unlink()
    a, b = take(prod), take(prod)
    assert a["errors"] == b["errors"] and a["canonical"] is None
    verdict, problems = dd.compare(a, b)
    assert verdict == "INCOMPLETE" and dd.EXIT[verdict] == 2


def test_compare_cli_exit_codes(prod, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(take(prod))); b.write_text(json.dumps(take(prod)))
    assert dd.main(["--compare", str(a), str(b)]) == 0
    bad = take(prod); bad["errors"] = {"neo4j": "x"}; bad["complete"] = False
    b.write_text(json.dumps(bad))
    assert dd.main(["--compare", str(a), str(b)]) == 2


def test_failed_verification_is_incomplete(prod):
    a, b = take(prod), take(prod)
    b["verify"] = {"graph_verify": {"ok": False}}
    assert dd.compare(a, b)[0] == "INCOMPLETE"


def test_busy_detection_is_read_only_and_does_not_create_lock_files(prod):
    assert dd.busy(prod) == [] and not (prod / ".heavy-work.lock").exists()
    (prod / "spool").mkdir(); (prod / "spool" / "job-1").mkdir()
    con = sqlite3.connect(prod / "pipeline.db"); con.execute("INSERT INTO sessions VALUES(1, NULL, 9e12, 0)"); con.commit(); con.close()
    reasons = dd.busy(prod, now=9e12)
    assert any("spool" in r for r in reasons) and any("capture" in r for r in reasons)


def test_held_heavy_lock_is_reported_busy(prod):
    import fcntl, os
    p = prod / ".heavy-work.lock"; p.write_text("")
    fd = os.open(p, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert any("heavy work" in r for r in dd.busy(prod))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
    assert dd.busy(prod) == []


def test_main_refuses_when_busy_without_interrupting(prod, monkeypatch, capsys):
    monkeypatch.setattr(config, "OUTPUT_DIR", prod)
    (prod / "spool").mkdir(); (prod / "spool" / "job-1").mkdir()
    assert dd.main([]) == 3 and "BUSY" in capsys.readouterr().out
    assert (prod / "spool" / "job-1").exists()


def test_chroma_lock_log_rows_are_reported_but_not_a_difference(prod):
    con = sqlite3.connect(prod / "chroma" / "chroma.sqlite3")
    con.execute("CREATE TABLE acquire_write(id INTEGER PRIMARY KEY, lock_status INTEGER NOT NULL)"); con.commit()
    a = take(prod)
    con.execute("INSERT INTO acquire_write(lock_status) VALUES(1)"); con.commit(); con.close()
    verdict, notes = dd.compare(a, take(prod))
    assert verdict == "IDENTICAL" and any("bookkeeping" in n and "rows" in n for n in notes)


def test_bookkeeping_table_with_another_schema_fails_closed(prod):
    con = sqlite3.connect(prod / "chroma" / "chroma.sqlite3")
    con.execute("CREATE TABLE acquire_write(id INTEGER PRIMARY KEY, payload TEXT)"); con.commit(); con.close()
    d = take(prod)
    assert not d["complete"] and "lock-log schema" in d["errors"]["chroma"]
