"""scripts/neo4j.sh: syntax, first/repeated install, credential safety, default command, password generation.
Uses a stub Neo4j home (bin/neo4j, bin/neo4j-admin) in a temp dir: no download, no real database."""
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "neo4j.sh"

@pytest.fixture
def env(tmp_path):
    home = tmp_path / "neo4j"
    (home / "bin").mkdir(parents=True)
    log = tmp_path / "calls.log"
    data = tmp_path / "data"
    # neo4j-admin stub: records the call and initializes the database like the real one (auth.ini)
    (home / "bin" / "neo4j-admin").write_text(
        f'#!/usr/bin/env bash\necho "admin $*" >> "{log}"\nmkdir -p "{data}/data/dbms"\n'
        f'echo initialized > "{data}/data/dbms/auth.ini"\n')
    (home / "bin" / "neo4j").write_text(f'#!/usr/bin/env bash\necho "neo4j $* conf=$NEO4J_CONF" >> "{log}"\n'
                                        f'echo "Neo4j is not running."\n')
    for f in (home / "bin").iterdir():
        f.chmod(0o755)
    e = {**os.environ, "NEO4J_HOME_DIR": str(home), "NEO4J_CONF_DIR": str(tmp_path / "conf"),
         "NEO4J_DATA_DIR": str(data), "NEO4J_ENV_FILE": str(tmp_path / ".env.local"), "JAVA_HOME": "/nonexistent"}
    return {"env": e, "log": log, "data": data, "envfile": tmp_path / ".env.local", "conf": tmp_path / "conf"}

def run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env["env"], capture_output=True, text=True)

def calls(env):
    return env["log"].read_text().splitlines() if env["log"].exists() else []

def creds(env):
    return dict(line.split("=", 1) for line in env["envfile"].read_text().splitlines() if "=" in line)

def test_shell_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0

def test_first_install_creates_config_and_credentials(env):
    r = run(env, "install")
    assert r.returncode == 0, r.stderr
    c = creds(env)
    assert set(c) == {"NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"}
    assert re.fullmatch(r"[0-9a-f]{32}", c["NEO4J_PASSWORD"])
    assert c["NEO4J_URI"] == "bolt://127.0.0.1:7697"
    assert stat.S_IMODE(env["envfile"].stat().st_mode) == 0o600
    assert calls(env) == [f"admin dbms set-initial-password {c['NEO4J_PASSWORD']}"]
    conf = (env["conf"] / "neo4j.conf").read_text()
    assert "server.bolt.listen_address=127.0.0.1:7697" in conf and "server.http.listen_address=127.0.0.1:7484" in conf
    assert "server.default_listen_address=127.0.0.1" in conf

def test_repeated_install_never_regenerates_or_resets(env):
    assert run(env, "install").returncode == 0
    before = env["envfile"].read_text()
    for _ in range(3):
        r = run(env, "install")
        assert r.returncode == 0 and "unchanged" in r.stdout
    assert env["envfile"].read_text() == before
    assert len([c for c in calls(env) if c.startswith("admin")]) == 1     # initial password set exactly once

def test_initialized_database_without_credentials_stops_and_writes_nothing(env):
    assert run(env, "install").returncode == 0
    env["envfile"].unlink()
    r = run(env, "install")
    assert r.returncode != 0 and "already initialized" in r.stderr
    assert not env["envfile"].exists()
    assert len([c for c in calls(env) if c.startswith("admin")]) == 1

def test_incomplete_credentials_are_refused(env):
    env["envfile"].write_text("NEO4J_URI=bolt://127.0.0.1:7697\nOTHER=1\n")
    r = run(env, "install")
    assert r.returncode != 0 and "missing: NEO4J_USER NEO4J_PASSWORD" in r.stderr
    assert env["envfile"].read_text() == "NEO4J_URI=bolt://127.0.0.1:7697\nOTHER=1\n"
    assert calls(env) == []

def test_interrupted_install_reuses_saved_credentials(env):
    env["envfile"].write_text("KEEP=me\nNEO4J_URI=bolt://127.0.0.1:7697\nNEO4J_USER=neo4j\nNEO4J_PASSWORD=" + "a" * 32 + "\n")
    r = run(env, "install")
    assert r.returncode == 0 and "reusing" in r.stdout
    assert calls(env) == ["admin dbms set-initial-password " + "a" * 32]
    assert creds(env)["KEEP"] == "me"

def test_new_credentials_preserve_other_lines(env):
    env["envfile"].write_text("UNRELATED=keep\n")
    assert run(env, "install").returncode == 0
    assert creds(env)["UNRELATED"] == "keep" and len(creds(env)["NEO4J_PASSWORD"]) == 32

def test_no_argument_runs_status(env):
    assert run(env, "install").returncode == 0
    r = run(env)
    assert r.returncode == 0 and "not running" in r.stdout
    assert calls(env)[-1].startswith("neo4j status conf=")

def test_password_generation_is_reliable_and_unique(env, tmp_path):
    pws = set()
    for i in range(20):                       # fresh state each time: exercises gen_password under pipefail
        e = {**env["env"], "NEO4J_DATA_DIR": str(tmp_path / f"d{i}"), "NEO4J_ENV_FILE": str(tmp_path / f"e{i}")}
        r = subprocess.run(["bash", str(SCRIPT), "install"], env=e, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        pws.add(Path(e["NEO4J_ENV_FILE"]).read_text().split("NEO4J_PASSWORD=")[1].strip())
    assert len(pws) == 20 and all(re.fullmatch(r"[0-9a-f]{32}", p) for p in pws)

def test_unknown_command_fails(env):
    r = run(env, "explode")
    assert r.returncode != 0 and "usage" in r.stderr