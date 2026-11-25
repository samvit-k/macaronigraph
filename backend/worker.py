"""Background processing with a crash-safe on-disk spool.

/api/process only *receives* a capture: the upload is written to output/spool/.incoming-<uuid>/, then —
atomically with respect to reconciliation — the Reel is marked `queued` and the directory is renamed to
output/spool/<ns>_<rand>/. One worker thread processes jobs oldest-first (ffmpeg → Whisper → OCR →
record → SQLite), so capturing the next Reel overlaps with processing the previous one.

Durability rules
  * identity (session_id, reel_id) always comes from meta.json, never from directory names;
  * mark-`queued` + publish happen under _publish_lock, which reconcile() also takes: reconciliation can
    never observe a Reel marked `queued` whose job is not yet visible (single backend process);
  * the built record is checkpointed to record.json BEFORE the SQLite write; the job (and its media) is
    deleted ONLY after the attempt is persisted. A DB failure keeps the job; retries reuse record.json;
  * after MAX_PERSIST_FAILURES, or if metadata is unreadable, the job is quarantined in spool/.dead/ and
    SQLite is reconciled immediately (no Reel stays `queued` without a live job);
  * quarantine retention: if record.json exists the Instagram audio/frames are deleted at once (the record
    holds the extracted text); if not, the media is the only recoverable copy and is kept until an
    explicit `purge-dead` — it is never deleted silently;
  * recovery is idempotent: half-written uploads are removed, jobs already persisted (crash after commit)
    are just cleaned up, other jobs resume, orphan `queued` marks are restored;
  * the worker loop catches everything; a failing job never blocks other due jobs.

CLI:  python -m backend.worker status | retry-dead | purge-dead [--older-than DAYS]
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from . import config, evalframes, jobs, ocr, record, transcribe

MAX_BACKLOG = 2              # uploads wait (backpressure) while more than this many jobs are spooled
BACKPRESSURE_TIMEOUT_S = 900
MAX_PERSIST_FAILURES = 3
RETRY_BACKOFF_S = 30
MEDIA = ("audio.webm", "frames")
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_publish_lock = threading.RLock()   # mark_queued + publish  vs  reconcile
_wake = threading.Event()
_stop = threading.Event()
_thread: threading.Thread | None = None
PostLookup = Callable[[str, str | None], dict | None]


class UnsafeJobError(ValueError):
    pass


class Overloaded(RuntimeError):
    pass


def spool_dir() -> Path:
    return config.OUTPUT_DIR / "spool"


def dead_dir() -> Path:
    return spool_dir() / ".dead"


def _check(v, what: str) -> str:
    if not isinstance(v, str) or not _SAFE.fullmatch(v):
        raise UnsafeJobError(f"unsafe {what}: {v!r}")
    return v


def pending_jobs() -> list[Path]:
    d = spool_dir()
    return sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")) if d.exists() else []


def job_identity(job: Path) -> tuple[str, str]:
    """(session_id, reel_id) from meta.json. Raises if missing/corrupt/unsafe."""
    meta = json.loads((job / "meta.json").read_text())
    return _check(meta.get("session_id"), "session_id"), _check(meta.get("reel_id"), "reel_id")


def _live_reel_ids() -> list[str]:
    ids = []
    for job in pending_jobs():
        try:
            ids.append(job_identity(job)[1])
        except Exception:  # noqa: BLE001
            continue
    return ids


def find_job(session_id: str | None, reel_id: str | None) -> Path | None:
    for job in pending_jobs():
        try:
            if job_identity(job) == (session_id, reel_id):
                return job
        except Exception:  # noqa: BLE001
            continue
    return None


def reconcile() -> int:
    """`queued` Reels with no live job → prior state. Serialized with publication. → Reels restored"""
    with _publish_lock:
        return jobs.restore_orphan_queued(_live_reel_ids())


def _publish(tmp: Path, rid: str, sid: str) -> tuple[Path, dict | None]:
    """Mark the Reel `queued` and make the job visible, as one step w.r.t. reconcile()."""
    with _publish_lock:
        snapshot = jobs.mark_queued(rid, sid)
        final = spool_dir() / f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        try:
            tmp.rename(final)                                   # atomic publication of a complete job
        except BaseException:
            jobs.revert_queued(rid, snapshot)                   # exact undo; the session keeps its lease
            raise
    return final, snapshot


def spool(cap: dict, audio_file, frames: list) -> Path:
    """Persist one capture as a job (see module docstring for the ordering guarantees)."""
    sid, rid = _check(cap.get("session_id"), "session_id"), _check(cap.get("reel_id"), "reel_id")
    spool_dir().mkdir(parents=True, exist_ok=True)
    tmp = spool_dir() / f".incoming-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        if audio_file is not None:
            with open(tmp / "audio.webm", "wb") as f:
                shutil.copyfileobj(audio_file, f)
        (tmp / "frames").mkdir()
        for i, fr in enumerate(frames):
            with open(tmp / "frames" / f"{i:04d}.jpg", "wb") as f:
                shutil.copyfileobj(fr, f)
        (tmp / "meta.json").write_text(json.dumps(cap, ensure_ascii=False))
        final, _ = _publish(tmp, rid, sid)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _wake.set()
    return final


def _shift(segments: list[dict], lead: float) -> None:
    for s in segments:
        s["start"], s["end"] = round(max(0.0, s["start"] - lead), 2), round(max(0.0, s["end"] - lead), 2)


def _missing_transcript(error: str) -> dict:
    return {"model": config.WHISPER_MODEL, "audio": None, "audio_class": "missing", "language": None, "text": "",
            "uncertain_text": "", "segments": [], "rejected_segments": [], "status": "error", "error": error}


def build(job: Path, lookup: PostLookup) -> dict:
    """Extraction for one job → record (not persisted). Extraction errors become a `processing_error`
    on the record (a failed, retryable attempt) instead of an exception."""
    cap = json.loads((job / "meta.json").read_text())
    source = cap.pop("source", "test")
    post = lookup(source, cap.get("reel_id"))
    transcript = ocr_result = None
    try:
        if cap.get("playback") not in record.FATAL_PLAYBACK:
            play_wall, rec_wall = cap.get("play_wall"), cap.get("rec_start_wall")
            lead = (play_wall - rec_wall) / 1000 if play_wall and rec_wall else 0.0
            cap["audio_lead_s"] = round(lead, 3)
            apath = job / "audio.webm"
            if apath.exists():
                cap["audio_bytes"] = apath.stat().st_size
                transcript = transcribe.transcribe_file(apath)
                _shift(transcript["segments"], lead)
                _shift(transcript["rejected_segments"], lead)
            else:
                transcript = _missing_transcript("no audio uploaded")
            walls = cap.pop("frame_walls", [])
            zero = play_wall or rec_wall or 0
            files = sorted((job / "frames").glob("*.jpg")) if (job / "frames").exists() else []
            batch = [(max(0.0, (w - zero) / 1000), p.read_bytes()) for w, p in zip(walls, files)]
            interval_s = (cap.get("frame_interval_ms") or 1000) / 1000
            if evalframes.enabled_for(cap.get("reel_id")):  # opt-in debug only; save() re-checks
                try:
                    cap["eval_frames_saved"] = bool(evalframes.save(cap["reel_id"], cap.get("session_id"),
                                                                    batch, interval_s, transcript))
                except Exception as e:  # noqa: BLE001
                    cap["eval_frames_saved"] = False
                    cap["eval_frames_error"] = f"{type(e).__name__}: {e}"
            md = (post or {}).get("metadata", {})
            ocr_result = ocr.ocr_frames(batch, interval_s=interval_s,
                                        transcript_segments=(transcript or {}).get("segments"),
                                        context_text=" ".join([md.get("caption", ""), *md.get("hashtags", [])]))
    except Exception as e:  # noqa: BLE001
        cap["processing_error"] = f"{type(e).__name__}: {e}"
        cap["processing_traceback"] = traceback.format_exc()[-2000:]
        ocr_result = None
    return record.build_record(post, cap, transcript, ocr_result)


def _summary(rec: dict, out: dict) -> dict:
    t, o = rec.get("transcript") or {}, rec.get("ocr") or {}
    return {"reel_id": rec["reel_id"], "status": rec["status"], "errors": rec["errors"],
            "state": out["state"], "attempt_id": out["attempt_id"], "duplicate": out["duplicate"],
            "audio_class": t.get("audio_class"), "transcript_chars": len(t.get("text", "")),
            "ocr_segments": len(o.get("segments", []))}


def _failures(job: Path) -> dict:
    """Failure bookkeeping. Corrupt bookkeeping counts as a failure (so it can still reach quarantine)
    and is due immediately — it can never park a job forever."""
    try:
        f = json.loads((job / "failures.json").read_text())
        return {"count": int(f["count"]), "last": float(f["last"]), "errors": list(f.get("errors", []))}
    except FileNotFoundError:
        return {"count": 0, "last": 0.0, "errors": []}
    except Exception as e:  # noqa: BLE001
        return {"count": 1, "last": 0.0, "errors": [f"corrupt failures.json: {type(e).__name__}"]}


VALID_STATUSES = {"ok", "ok_no_speech", "partial", "failed"}


def valid_checkpoint(job: Path) -> bool:
    """record.json parses into a recoverable extracted record with well-typed core fields (not merely:
    the file exists). Only then may quarantine drop the original media."""
    try:
        rec = json.loads((job / "record.json").read_text())
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(rec, dict):
        return False
    return (isinstance(rec.get("reel_id"), str) and _SAFE.fullmatch(rec["reel_id"]) is not None
            and isinstance(rec.get("session_id"), str) and bool(rec["session_id"])
            and rec.get("status") in VALID_STATUSES
            and isinstance(rec.get("errors"), list) and all(isinstance(e, str) for e in rec["errors"])
            and isinstance(rec.get("processed_at"), str)
            and isinstance(rec.get("capture"), dict)
            and (rec.get("transcript") is None or isinstance(rec["transcript"], dict))
            and (rec.get("ocr") is None or isinstance(rec["ocr"], dict)))


def _set_aside_corrupt_checkpoint(job: Path) -> bool:
    """A present-but-invalid record.json is kept as record.json.corrupt (evidence) so it is never reused."""
    p = job / "record.json"
    if p.exists() and not valid_checkpoint(job):
        p.rename(job / "record.json.corrupt")
        return True
    return False


def _quarantine(job: Path, reason: str) -> Path:
    """Move a job to spool/.dead, apply the retention policy, and reconcile SQLite immediately.
    Media is deleted only if a VALID checkpoint holds the extracted record; otherwise it is the only
    recoverable copy and is kept."""
    dead_dir().mkdir(parents=True, exist_ok=True)
    _set_aside_corrupt_checkpoint(job)
    has_record = valid_checkpoint(job)
    (job / "QUARANTINED.txt").write_text(json.dumps({
        "reason": reason, "at": time.time(), "record_checkpoint": has_record,
        "media": "deleted (extracted record kept)" if has_record else "kept: only recoverable copy"}, indent=1))
    target = dead_dir() / job.name
    job.rename(target)
    if has_record:  # the record already holds the extracted text → don't retain Instagram media
        for m in MEDIA:
            p = target / m
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    try:
        reconcile()
    except Exception:  # noqa: BLE001 — DB down: recover()/next quarantine reconciles later
        pass
    return target


def process_job(job: Path, lookup: PostLookup) -> dict | None:
    """Build (or reuse the checkpointed) record, persist it, and only then delete the job.
    → summary, or None if persistence failed (job kept for retry, or quarantined)."""
    checkpoint = job / "record.json"
    if checkpoint.exists():
        rec = json.loads(checkpoint.read_text())          # extraction already done: don't redo it
    else:
        rec = build(job, lookup)
        tmp = checkpoint.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.rename(checkpoint)
    try:
        out = jobs.record_attempt(rec)
    except Exception as e:  # noqa: BLE001 — DB failure: keep the job and its media
        f = _failures(job)
        f.update(count=f["count"] + 1, last=time.time(), errors=(f["errors"] + [f"{type(e).__name__}: {e}"])[-5:])
        (job / "failures.json").write_text(json.dumps(f))
        if f["count"] >= MAX_PERSIST_FAILURES:
            _quarantine(job, f"persist failed {f['count']}x: {f['errors']}")
        return None
    shutil.rmtree(job, ignore_errors=True)                # persisted → media can go
    return _summary(rec, out)


def _due(job: Path) -> bool:
    f = _failures(job)
    return f["count"] == 0 or time.time() - f["last"] >= RETRY_BACKOFF_S


def run_once(lookup: PostLookup) -> bool:
    """Process the oldest *due* job (jobs in back-off are skipped, never blocking others). Never raises."""
    for job in pending_jobs():
        try:
            if not _due(job):
                continue
            try:
                job_identity(job)
            except Exception as e:  # noqa: BLE001
                _quarantine(job, f"unreadable job metadata: {type(e).__name__}: {e}")
                return True
            process_job(job, lookup)
        except Exception as e:  # noqa: BLE001 — e.g. corrupt record.json: quarantine, keep going
            try:
                if job.exists():
                    _quarantine(job, f"worker error: {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
            except Exception:  # noqa: BLE001
                pass
        return True
    return False


def drain(lookup: PostLookup, max_jobs: int | None = None) -> int:
    n = 0
    while (max_jobs is None or n < max_jobs) and run_once(lookup):
        n += 1
    return n


def recover() -> dict:
    """Startup recovery after a crash/restart (call before accepting uploads). Idempotent."""
    d = spool_dir()
    partial = list(d.glob(".incoming-*")) if d.exists() else []
    for p in partial:
        shutil.rmtree(p, ignore_errors=True)
    resumed = committed = quarantined = 0
    for job in pending_jobs():
        try:
            sid, rid = job_identity(job)
        except Exception as e:  # noqa: BLE001
            _quarantine(job, f"unreadable job metadata at recovery: {type(e).__name__}: {e}")
            quarantined += 1
            continue
        if jobs.existing_attempt(sid, rid):   # crash after commit: already persisted → just clean up
            shutil.rmtree(job, ignore_errors=True)
            committed += 1
            continue
        jobs.mark_queued(rid, sid)
        resumed += 1
    return {"partial_uploads_removed": len(partial), "jobs_resumed": resumed,
            "already_committed_cleaned": committed, "quarantined": quarantined, "orphans_restored": reconcile()}


def _loop(lookup: PostLookup) -> None:
    while not _stop.is_set():
        try:
            handled = run_once(lookup)
        except Exception:  # noqa: BLE001 — the thread must survive
            handled = False
        if not handled:
            _wake.wait(timeout=2.0)
            _wake.clear()


def start(lookup: PostLookup) -> dict:
    global _thread
    info = recover()
    if _thread is None or not _thread.is_alive():
        _stop.clear()
        _thread = threading.Thread(target=_loop, args=(lookup,), name="reel-worker", daemon=True)
        _thread.start()
    return info


def stop() -> None:
    _stop.set()
    _wake.set()


def alive() -> bool:
    return _thread is not None and _thread.is_alive()


def ensure_capacity(timeout_s: float | None = None, poll_s: float = 1.0) -> int:
    """Wait while the backlog exceeds MAX_BACKLOG; if still excessive after `timeout_s`, refuse
    (Overloaded): the caller must not accept the upload. → backlog when capacity is available"""
    timeout_s = BACKPRESSURE_TIMEOUT_S if timeout_s is None else timeout_s
    deadline = time.time() + timeout_s
    while (n := len(pending_jobs())) > MAX_BACKLOG:
        if time.time() >= deadline:
            raise Overloaded(f"{n} jobs still queued after {timeout_s:.0f}s (max {MAX_BACKLOG})")
        time.sleep(poll_s)
    return n


# ---------------------------------------------------------------- quarantine maintenance (explicit only)

def dead_jobs() -> list[dict]:
    out = []
    for p in sorted(dead_dir().iterdir()) if dead_dir().exists() else []:
        info = {}
        try:
            info = json.loads((p / "QUARANTINED.txt").read_text())
        except Exception:  # noqa: BLE001
            pass
        out.append({"job": p.name, "has_media": any((p / m).exists() for m in MEDIA),
                    "has_record": (p / "record.json").exists(), **info})
    return out


def retry_dead() -> int:
    """Move quarantined jobs that still have readable metadata back into the spool."""
    n = 0
    for p in sorted(dead_dir().iterdir()) if dead_dir().exists() else []:
        try:
            sid, rid = job_identity(p)
        except Exception:  # noqa: BLE001
            continue
        for f in ("failures.json", "QUARANTINED.txt"):
            (p / f).unlink(missing_ok=True)
        _set_aside_corrupt_checkpoint(p)   # never reuse a corrupt checkpoint: rebuild from media instead
        with _publish_lock:
            jobs.mark_queued(rid, sid)
            p.rename(spool_dir() / p.name)
        n += 1
    _wake.set()
    return n


def purge_dead(older_than_days: float) -> int:
    """Explicitly delete quarantined media older than N days (keeps record.json + diagnostics)."""
    n, cutoff = 0, time.time() - older_than_days * 86400
    for p in sorted(dead_dir().iterdir()) if dead_dir().exists() else []:
        if p.stat().st_mtime <= cutoff:
            for m in MEDIA:
                q = p / m
                if q.exists():
                    shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
                    n += 1
    return n


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "status"), sys.argv[2:]
    if cmd == "status":
        print(json.dumps({"pending_jobs": [p.name for p in pending_jobs()], "dead": dead_jobs()}, indent=1))
    elif cmd == "retry-dead":
        print(f"requeued {retry_dead()} quarantined job(s)")
    elif cmd == "purge-dead":
        days = float(args[args.index("--older-than") + 1]) if "--older-than" in args else 7.0
        print(f"deleted media from {purge_dead(days)} item(s) older than {days} days")
    else:
        sys.exit(__doc__)
