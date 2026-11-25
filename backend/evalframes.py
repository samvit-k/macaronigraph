"""OPT-IN debug store of sampled JPEG frames for OCR evaluation. Disabled by default.

Only Reels explicitly listed in output/eval_frames/selection.json (max MAX_SELECTED) have their *sampled
frames* (never video, never audio) written to output/eval_frames/<reel_id>/<session_id>/, with the frame
timestamps and transcript segments needed for offline OCR tuning. Everything stays on this machine
(output/ is git-ignored). `purge` deletes all of it and disables the mode.

CLI:  python -m backend.evalframes enable <reel_id>... | status | purge
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from . import config

MAX_SELECTED = 5
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # Instagram shortcodes, test ids, session ids

class UnsafeIdError(ValueError):
    pass

def eval_dir() -> Path:
    return config.OUTPUT_DIR / "eval_frames"

def _check_id(value: str | None, what: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise UnsafeIdError(f"unsafe {what}: {value!r}")
    return value

def _inside(path: Path) -> Path:
    """Resolve and assert the path stays inside eval_dir() (defence in depth after _check_id)."""
    root, p = eval_dir().resolve(), path.resolve()
    if root != p and root not in p.parents:
        raise UnsafeIdError(f"path escapes {root}: {p}")
    return p

def selection() -> list[str]:
    f = eval_dir() / "selection.json"
    if not f.is_file():
        return []
    ids = json.loads(f.read_text()).get("reel_ids", [])
    return [i for i in ids if isinstance(i, str) and _SAFE_ID.fullmatch(i)][:MAX_SELECTED]

def enabled_for(reel_id: str | None) -> bool:
    return bool(reel_id) and reel_id in selection()

def save(reel_id: str, session_id: str, frames: list[tuple[float, bytes]], interval_s: float,
         transcript: dict | None) -> Path | None:
    """Write frames ONLY if reel_id is explicitly selected; otherwise do nothing and return None."""
    if not enabled_for(reel_id):
        return None
    d = _inside(eval_dir() / _check_id(reel_id, "reel_id") / _check_id(session_id, "session_id"))
    d.mkdir(parents=True, exist_ok=True)
    names = []
    for i, (t, data) in enumerate(sorted(frames, key=lambda f: f[0])):
        name = f"{i:04d}_{float(t):07.2f}.jpg"
        (d / name).write_bytes(data)
        names.append({"file": name, "t": t})
    (d / "meta.json").write_text(json.dumps({
        "reel_id": reel_id, "session_id": session_id, "interval_s": interval_s, "frames": names,
        "transcript_segments": (transcript or {}).get("segments", []),
        "transcript_text": (transcript or {}).get("text", "")}, ensure_ascii=False, indent=1))
    return d

def load(reel_id: str) -> tuple[list[tuple[float, bytes]], dict]:
    """Latest saved capture for a Reel → (frames, meta)."""
    base = _inside(eval_dir() / _check_id(reel_id, "reel_id"))
    runs = sorted(base.glob("*/meta.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no eval frames for {reel_id}")
    meta = json.loads(runs[-1].read_text())
    return [(f["t"], (runs[-1].parent / f["file"]).read_bytes()) for f in meta["frames"]], meta

def enable(reel_ids: list[str]) -> list[str]:
    ids = list(dict.fromkeys(_check_id(i, "reel_id") for i in reel_ids))
    if not ids or len(ids) > MAX_SELECTED:
        raise ValueError(f"select between 1 and {MAX_SELECTED} Reels (got {len(ids)})")
    eval_dir().mkdir(parents=True, exist_ok=True)
    (eval_dir() / "selection.json").write_text(json.dumps({"reel_ids": ids}, indent=1))
    return ids

def purge() -> int:
    d = eval_dir()
    n = sum(1 for _ in d.rglob("*.jpg")) if d.exists() else 0
    shutil.rmtree(d, ignore_errors=True)
    return n

if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "status"), sys.argv[2:]
    if cmd == "enable":
        print(f"eval frames ENABLED for {enable(args)} → {eval_dir()}")
    elif cmd == "status":
        sel = selection()
        print(f"enabled for: {sel or 'nothing (disabled)'}")
        for rid in sel:
            print(f"  {rid}: {sum(1 for _ in (eval_dir() / rid).rglob('*.jpg'))} frames stored")
    elif cmd == "purge":
        print(f"deleted {purge()} frames and the selection; eval mode disabled")
    else:
        sys.exit(__doc__)