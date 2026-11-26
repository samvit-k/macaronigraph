"""scripts/service.sh: plist contents, install/uninstall through a fake launchctl, and refusing a backend already running."""
import os
import plistlib
import signal
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from tests.helpers import free_port

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service.sh"
LABEL = "com.macaronigraph.backend"

FAKE_LAUNCHCTL = """#!/usr/bin/env bash
echo "$*" >> "{log}"
case "$1" in
  print) [[ -f "{marker}" ]] ;;
  bootstrap) touch "{marker}"
    [[ -n "{serve}" ]] && {{ nohup "{python}" -c "{serve}" >/dev/null 2>&1 & echo $! > "{pid}"; }}
    exit 0 ;;
  bootout) rm -f "{marker}" ;;
esac
"""

SERVE = ("import http.server as h; R = type('R', (h.BaseHTTPRequestHandler,), {{'do_GET': lambda s: (s.send_response(200), "
         "s.end_headers()), 'log_message': lambda *a: None}}); h.HTTPServer(('127.0.0.1', {port}), R).serve_forever()")

def make_env(tmp_path, serve=False):
    port = free_port()
    log, marker, pid = tmp_path / "calls.log", tmp_path / "loaded", tmp_path / "server.pid"
    uvicorn = tmp_path / "venv" / "uvicorn"
    uvicorn.parent.mkdir()
    uvicorn.write_text("#!/bin/sh\n")
    uvicorn.chmod(0o755)
    fake = tmp_path / "launchctl"
    fake.write_text(FAKE_LAUNCHCTL.format(log=log, marker=marker, pid=pid, python=sys.executable,
                                          serve=SERVE.format(port=port) if serve else ""))
    fake.chmod(0o755)
    env = {**os.environ, "LAUNCHCTL": str(fake), "MACARONIGRAPH_AGENTS_DIR": str(tmp_path / "agents"), "PORT": str(port),
           "SERVICE_UVICORN": str(uvicorn), "SERVICE_LOG": str(tmp_path / "logs" / "backend.log"), "SERVICE_WAIT_S": "15"}
    return {"env": env, "log": log, "pid": pid, "port": port, "uvicorn": uvicorn,
            "plist": tmp_path / "agents" / f"{LABEL}.plist", "tmp": tmp_path}

@pytest.fixture
def env(tmp_path):
    e = make_env(tmp_path, serve=True)
    yield e
    if e["pid"].exists():
        os.kill(int(e["pid"].read_text()), signal.SIGTERM)

def run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env["env"], capture_output=True, text=True, timeout=60)

def calls(env):
    return [line.split()[0] for line in env["log"].read_text().splitlines()] if env["log"].exists() else []

def test_shell_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0

def test_install_writes_a_loopback_plist_and_waits_for_health(env):
    r = run(env, "install")
    assert r.returncode == 0, r.stderr
    assert f"running: http://127.0.0.1:{env['port']}/workspace/" in r.stdout
    p = plistlib.loads(env["plist"].read_bytes())
    assert p["Label"] == LABEL and p["WorkingDirectory"] == str(ROOT)
    assert p["ProgramArguments"] == [str(env["uvicorn"]), "backend.app:app", "--host", "127.0.0.1", "--port", str(env["port"])]
    assert p["RunAtLoad"] is True and p["KeepAlive"] == {"SuccessfulExit": False}
    assert p["StandardOutPath"] == p["StandardErrorPath"] == env["env"]["SERVICE_LOG"]
    assert calls(env) == ["print", "bootstrap"]
    assert "installed=yes loaded=yes" in run(env, "status").stdout

def test_reinstall_reloads_instead_of_refusing_its_own_backend(env):
    assert run(env, "install").returncode == 0
    os.kill(int(env["pid"].read_text()), signal.SIGTERM)
    r = run(env, "install")
    assert r.returncode == 0, r.stderr
    assert calls(env) == ["print", "bootstrap", "print", "bootout", "bootstrap"]

def test_uninstall_unloads_and_removes_the_plist(env):
    assert run(env, "install").returncode == 0
    r = run(env, "uninstall")
    assert r.returncode == 0 and "removed" in r.stdout
    assert not env["plist"].exists()
    assert calls(env)[-2:] == ["print", "bootout"]
    assert "not installed" in run(env, "uninstall").stdout

def test_refuses_when_another_backend_already_answers(tmp_path):
    e = make_env(tmp_path)

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", e["port"]), Health)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        r = run(e, "install")
    finally:
        server.shutdown()
    assert r.returncode == 1 and "already running" in r.stderr
    assert not e["plist"].exists() and calls(e) == ["print"]

def test_reports_a_backend_that_never_answers(tmp_path):
    e = make_env(tmp_path)
    e["env"]["SERVICE_WAIT_S"] = "1"
    r = run(e, "install")
    assert r.returncode == 1 and "did not answer" in r.stderr and "Privacy & Security" in r.stderr

def test_requires_setup_first(env):
    env["env"]["SERVICE_UVICORN"] = str(env["tmp"] / "missing" / "uvicorn")
    r = run(env, "install")
    assert r.returncode == 1 and "setup.sh" in r.stderr and calls(env) == []

def test_paths_with_xml_characters_are_escaped(env):
    odd = env["tmp"] / "a & <b>" / "uvicorn"
    odd.parent.mkdir()
    odd.write_text("#!/bin/sh\n")
    odd.chmod(0o755)
    env["env"]["SERVICE_UVICORN"] = str(odd)
    assert run(env, "install").returncode == 0
    assert plistlib.loads(env["plist"].read_bytes())["ProgramArguments"][0] == str(odd)