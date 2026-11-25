"""Graph CLI dispatch: argument checks, exit codes and store cleanup, with Neo4j and the queries stubbed."""
import pytest

from backend.graph import __main__ as cli
from backend.graph import queries as q
from backend.graph.store import GraphOffline

class Store:
    closed = 0

    def close(self):
        Store.closed += 1

    def run(self, cypher, **kw):
        return []

@pytest.fixture(autouse=True)
def stub(monkeypatch):
    Store.closed = 0
    monkeypatch.setattr(cli, "Neo4jStore", Store)
    monkeypatch.setattr(q, "status", lambda store: {"reels": 1})
    monkeypatch.setattr(q, "reel_concepts", lambda store, rid: {"reel_id": rid})

@pytest.mark.parametrize("argv", [["g", "nope"], ["g", "reel"], ["g", "between", "a"], ["g", "review", "x", "maybe"]])
def test_unknown_commands_and_missing_arguments_print_usage(argv, capsys):
    assert cli.main(argv) == 2
    assert "python -m backend.graph" in capsys.readouterr().out
    assert Store.closed == 1

def test_default_command_is_status(capsys):
    assert cli.main(["g"]) == 0
    assert '"reels": 1' in capsys.readouterr().out

def test_command_arguments_are_passed_through(capsys):
    assert cli.main(["g", "reel", "R1"]) == 0
    assert '"reel_id": "R1"' in capsys.readouterr().out

def test_verify_failure_and_missing_assertion_exit_1(monkeypatch, capsys):
    monkeypatch.setattr(q, "verify", lambda store: {"ok": False})
    assert cli.main(["g", "verify"]) == 1
    assert cli.main(["g", "review", "missing", "accept"]) == 1
    assert "no assertion 'missing'" in capsys.readouterr().err

def test_graph_offline_exits_3_and_closes_the_store(monkeypatch):
    def offline(store):
        raise GraphOffline("stopped")
    monkeypatch.setattr(q, "status", offline)
    assert cli.main(["g", "status"]) == 3
    assert Store.closed == 1