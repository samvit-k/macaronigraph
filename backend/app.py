"""FastAPI backend: capture processing (Phase 1) + SQLite job tracking (Phase 2) + semantic search (Phase 3)
+ knowledge graph (Phase 4A) + retrieval and local answers, chat page at /app (Phase 4B)
+ MacaroniGraph workspace at /workspace/ and local welcome page at /welcome (Phase 5).

Run:  .venv/bin/uvicorn backend.app:app --port 8765     (uvicorn binds 127.0.0.1 by default; keep it local)
"""
from __future__ import annotations

import io
import json
import sys
import uuid
import zipfile
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, evalframes, indexer, jobs, metadata, worker

SOURCES = {"test": config.TEST_POSTS_PATH, "instagram": config.INSTAGRAM_POSTS_PATH}


@lru_cache(maxsize=4)
def _posts(source: str) -> dict[str, dict]:
    path = SOURCES[source]
    if not Path(path).is_file():
        raise HTTPException(404, f"source file not found: {path}")
    return {p["reel_id"]: p for p in metadata.load_posts(Path(path)) if p["reel_id"]}


def _lookup(source: str, reel_id: str | None) -> dict | None:
    return _posts(source).get(reel_id) if source in SOURCES else None


@asynccontextmanager
async def lifespan(_app):
    if config.ASYNC_PROCESSING:  # crash recovery runs before any upload is accepted
        print(f"[worker] started; recovery: {worker.start(_lookup)}", file=sys.stderr)
        indexer.start()  # indexes only while capture + processing are idle (INDEX_AUTO); manual sync anytime
        _graph_call("start")  # knowledge graph: optional, idle-gated; Neo4j offline never affects the backend
        _answers_call("start")  # Phase 4B: deferred answer generation (expires queued turns of a previous run)
    yield
    if config.ASYNC_PROCESSING:
        _answers_call("stop")
        _graph_call("stop")
        indexer.stop()
    worker.stop()
    try:
        from . import models
        models.shutdown()                   # free the shared local LLM (waits for a running generation)
    except Exception as e:  # noqa: BLE001
        print(f"[models] shutdown failed: {type(e).__name__}: {e}", file=sys.stderr)


def _answers_call(fn: str):
    try:
        from . import ask
        return getattr(ask, fn)()
    except Exception as e:  # noqa: BLE001 — answers are optional; retrieval and the backend keep working
        print(f"[answers] {fn} failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _graph_call(fn: str):
    try:
        from .graph import sync as graph_sync
        return getattr(graph_sync, fn)()
    except Exception as e:  # noqa: BLE001 — the graph is optional
        print(f"[graph] {fn} failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


app = FastAPI(title="MacaroniGraph", version=config.PIPELINE_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=r"chrome-extension://.*", allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"ok": True, "whisper_model": config.WHISPER_MODEL, "ffmpeg": config.FFMPEG,
            "tesseract": config.TESSERACT, "output": str(config.OUTPUT_JSONL),
            "sources": {k: Path(v).is_file() for k, v in SOURCES.items()},
            "async_processing": config.ASYNC_PROCESSING, "worker_alive": worker.alive(),
            "backlog": len(worker.pending_jobs()), "quarantined": len(worker.dead_jobs()),
            "index": _index_summary()}


def _index_summary() -> dict:
    try:
        return {"active": indexer.active_collection(), "build": (indexer._meta_json(indexer.K_BUILD) or {}).get("state"),
                "running": indexer._state["running"], "busy": indexer._state["busy"]}
    except Exception as e:  # noqa: BLE001 — health must never fail because of the index
        return {"error": str(e)}


@app.get("/api/session")
def session(source: str = "test", limit: int = 3, offset: int = 0, mode: str = "pending"):
    """Create a processing session: lease the next Reels from the job store and return their URLs.
    mode=pending → only pending / retryable Reels (resume); mode=all → reprocess everything."""
    if source not in SOURCES:
        raise HTTPException(400, f"source must be one of {list(SOURCES)}")
    if mode not in ("pending", "all", "eval"):
        raise HTTPException(400, "mode must be 'pending', 'all' or 'eval'")
    jobs.import_posts(source, list(_posts(source).values()))  # idempotent; never resets job state
    sid = f"{source}-{uuid.uuid4().hex[:8]}"
    ids = evalframes.selection() if mode == "eval" else None  # empty selection → empty queue
    out = {"session_id": sid, "source": source, "mode": mode,
           "items": jobs.create_session(sid, source, limit, offset, mode, reel_ids=ids)}
    # Manifest of what this run is expected to process (used to verify completeness afterwards).
    sessions = config.OUTPUT_DIR / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / f"{sid}.json").write_text(json.dumps(out, indent=1))
    return out


@app.post("/api/session/{session_id}/heartbeat")
def session_heartbeat(session_id: str):
    return {"ok": jobs.heartbeat(session_id)}


@app.post("/api/session/{session_id}/finish")
def session_finish(session_id: str, reason: str = "finished"):
    return {"ok": True, "released": jobs.finish_session(session_id, reason)}


@app.get("/api/session/{session_id}/status")
def session_status(session_id: str):
    """Capture finished? Uploads still queued/processing? (the extension waits for queued == 0)"""
    return {**jobs.session_progress(session_id), "backlog": len(worker.pending_jobs())}


@app.get("/api/status")
def job_status(source: str | None = None):
    return {**jobs.status(source), "backlog": len(worker.pending_jobs()), "quarantined": len(worker.dead_jobs())}


@app.post("/api/process")
def process(meta: str = Form(...), audio: UploadFile | None = File(None),
            frames: list[UploadFile] = File(default_factory=list)):
    """One captured item (audio webm/opus + sampled JPEG frames + capture metadata).
    Async mode: spooled and processed by the background worker (response: status "queued"); 503 if the
    backlog stays excessive. Sync mode: processed inline. Spooled media is deleted once persisted."""
    try:
        cap = json.loads(meta)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"meta is not JSON: {e}")
    uploads = [u for u in [audio, *frames] if u is not None]
    try:
        cap.setdefault("session_id", f"nosession-{uuid.uuid4().hex[:8]}")
        sid, rid = cap.get("session_id"), cap.get("reel_id")
        prior = jobs.existing_attempt(sid, rid)
        if prior:  # repeated upload for the same session+Reel: do not reprocess or double count
            return {"reel_id": rid, "status": prior["status"], "errors": json.loads(prior["errors"]),
                    "duplicate": True, "attempt_id": prior["attempt_id"]}
        if worker.find_job(sid, rid):  # same upload already waiting in the spool
            return {"reel_id": rid, "status": "queued", "errors": [], "duplicate": True, "queued": True}
        if config.ASYNC_PROCESSING:
            try:
                backlog = worker.ensure_capacity()
            except worker.Overloaded as e:  # refuse BEFORE accepting anything: nothing is spooled
                raise HTTPException(503, f"backend overloaded: {e}")
        try:
            job = worker.spool(cap, audio.file if audio else None, [f.file for f in frames])
        except worker.UnsafeJobError as e:
            raise HTTPException(400, str(e))
    finally:
        for up in uploads:
            up.file.close()
    if config.ASYNC_PROCESSING:
        return {"reel_id": rid, "status": "queued", "errors": [], "queued": True, "duplicate": False,
                "backlog": backlog + 1}
    out = worker.process_job(job, _lookup)  # synchronous mode: same code path, inline
    if out is None:
        raise HTTPException(500, "capture spooled but could not be persisted; it will be retried")
    return out


@app.get("/api/results")
def results(session_id: str | None = None):
    """Attempt history (from SQLite), optionally for one session."""
    with jobs.connect() as con:
        q = "SELECT attempt_id, reel_id, session_id, processed_at, status, outcome, errors FROM attempts"
        rows = con.execute(q + (" WHERE session_id=?" if session_id else "") + " ORDER BY attempt_id",
                           (session_id,) if session_id else ()).fetchall()
    return [{**dict(r), "errors": json.loads(r["errors"] or "[]")} for r in rows]


# ---------- Phase 3: semantic search ----------

@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=1000), k: int = Query(10, ge=1, le=50),
           source: str = "instagram", types: str | None = None, group: str = "reel"):
    """Semantic search over indexed Reels. types: comma list of caption, transcript, ocr (= all on-screen
    kinds), ocr_title, ocr_screen, ocr_fragment. group: reel (best passages per Reel) | none (raw passages)."""
    if group not in ("reel", "none"):
        raise HTTPException(400, "group must be 'reel' or 'none'")
    kinds = [t.strip() for t in types.split(",") if t.strip()] if types else None
    if kinds and (bad := [t for t in kinds if t not in indexer.TYPE_GROUPS]):
        raise HTTPException(400, f"unknown types {bad}; use {sorted(indexer.TYPE_GROUPS)}")
    try:
        return indexer.search(q, k=k, source=source, types=kinds, group=group)
    except indexer.IndexNotReady as e:
        raise HTTPException(503, str(e))


@app.get("/api/index/status")
def index_status():
    return indexer.status()


@app.post("/api/index/sync")
def index_sync(rebuild: bool = False, wait: bool = False):
    """Manual sync (always allowed, also during capture). wait=1 runs it inline and returns the result."""
    try:
        return indexer.request_sync(rebuild=rebuild, wait=wait)
    except indexer.WriterBusy as e:
        raise HTTPException(409, str(e))


# ---------- Phase 4A: knowledge graph (optional; 503 when Neo4j is offline) ----------

def _graph(fn, *args, **kw):
    from .graph import queries as gq
    from .graph.store import GraphOffline, Neo4jStore
    store = Neo4jStore()
    try:
        return getattr(gq, fn)(store, *args, **kw)
    except GraphOffline as e:
        raise HTTPException(503, f"knowledge graph offline: {e}")
    finally:
        store.close()


@app.get("/api/graph/status")
def graph_status():
    from .graph import queries as gq
    return gq.status()


@app.post("/api/graph/sync")
def graph_sync(force: bool = False, wait: bool = False):
    """Manual graph sync. Cached extractions are always written; new LLM extraction runs only while capture and
    processing are idle, unless force=1. wait=1 runs it now (or reports busy)."""
    from .graph import sync as gs
    return gs.request_sync(force=force, wait=wait)


@app.get("/api/graph/request/{request_id}")
def graph_request(request_id: int):
    from .graph import sync as gs
    r = gs.request_status(request_id)
    if r is None:
        raise HTTPException(404, "unknown request id")
    return r


@app.get("/api/graph/verify")
def graph_verify():
    return _graph("verify")


@app.get("/api/graph/reel/{reel_id}")
def graph_reel(reel_id: str):
    return _graph("reel_concepts", reel_id)


@app.get("/api/graph/concept")
def graph_concept(name: str = Query(..., min_length=1)):
    return _graph("concept_reels", name)


@app.get("/api/graph/shared")
def graph_shared(min_reels: int = Query(2, ge=1), limit: int = Query(50, ge=1, le=500)):
    return _graph("shared", min_reels, limit)


@app.get("/api/graph/between")
def graph_between(a: str = Query(..., min_length=1), b: str = Query(..., min_length=1), include_rejected: bool = False):
    return _graph("between", a, b, include_rejected=include_rejected)


# ---------- Phase 5: read-only views for the MacaroniGraph workspace (additive; see backend/graph/view.py) ----------

def _view(fn, *args, **kw):
    from .graph import view as gv
    from .graph.store import GraphOffline, Neo4jStore
    store = Neo4jStore()
    try:
        return getattr(gv, fn)(store, *args, **kw)
    except GraphOffline as e:
        raise HTTPException(503, f"knowledge graph offline: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        store.close()


@app.get("/api/graph/overview")
def graph_overview(min_reels: int = Query(2, ge=1, le=50), limit_concepts: int = Query(150, ge=1, le=1000),
                   reel_offset: int = Query(0, ge=0), reel_limit: int = Query(200, ge=1, le=1000)):
    return _view("overview", min_reels=min_reels, limit_concepts=limit_concepts, reel_offset=reel_offset,
                 reel_limit=reel_limit)


@app.get("/api/graph/neighborhood")
def graph_neighborhood(kind: str = Query(..., pattern="^(reel|concept)$"), id: str = Query(..., min_length=1),
                       limit: int = Query(200, ge=1, le=1000)):
    r = _view("neighborhood", kind, id, limit=limit)
    if not r.get("found"):
        raise HTTPException(404, f"unknown {kind}: {id}")
    return r


@app.get("/api/graph/search")
def graph_search(q: str = Query(..., min_length=1, max_length=200), limit: int = Query(20, ge=1, le=100)):
    return _view("search", q, limit=limit)


@app.get("/api/graph/evidence")
def graph_evidence(a: str = Query(..., min_length=1), b: str = Query(..., min_length=1), include_rejected: bool = False):
    return _graph("evidence", a, b, include_rejected=include_rejected)


# ---------- Phase 4B: retrieval + local answers ----------

class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None
    mode: str = "hybrid"
    generate: bool = True
    options: dict | None = None


_ASK_OPTIONS = {"co_mention", "relations", "max_sources"}      # public knobs; tuning parameters stay internal


@app.post("/api/ask")
def ask_endpoint(body: AskBody):
    """Vector, graph or hybrid retrieval (always returned) + a validated local answer, or a deferred request id
    while capture/processing are active. Graph unavailable → vector retrieval, reported in `fallback`."""
    from . import ask, rag
    opts = {k: v for k, v in (body.options or {}).items() if k in _ASK_OPTIONS}
    if "max_sources" in opts:
        opts["max_sources"] = max(1, min(int(opts["max_sources"]), 20))
    try:
        return ask.ask(body.question, conversation_id=body.conversation_id, mode=body.mode, options=opts,
                       generate=body.generate)
    except ask.AskError as e:
        raise HTTPException(400, str(e))
    except rag.RetrievalUnavailable as e:
        raise HTTPException(503, str(e))


@app.get("/api/ask/{request_id}")
def ask_request(request_id: int):
    from . import ask
    r = ask.request_status(request_id)
    if r is None:
        raise HTTPException(404, "unknown request id")
    return r


@app.get("/api/conversations")
def conversations_list():
    from . import conversations
    return conversations.list_conversations()


@app.get("/api/conversations/{conversation_id}")
def conversation_get(conversation_id: str):
    from . import ask, conversations
    c = conversations.get(conversation_id)
    if c is None:
        raise HTTPException(404, "unknown conversation")
    return {**{k: v for k, v in c.items() if k != "turns"}, "turns": [ask.response(t) for t in c["turns"]]}


@app.delete("/api/conversations")
def conversations_clear():
    """Phase 5 (additive): delete ALL conversations, including the one currently open. Reel data is untouched."""
    from . import conversations
    return {"deleted": conversations.delete_all()}


@app.delete("/api/conversations/{conversation_id}")
def conversation_delete(conversation_id: str):
    from . import conversations
    if not conversations.delete(conversation_id):
        raise HTTPException(404, "unknown conversation")
    return {"deleted": conversation_id}


@app.get("/app")
@app.get("/app/{path:path}")
def legacy_app_redirect(path: str = ""):
    return RedirectResponse("/workspace/")


# ---------- Phase 5: MacaroniGraph workspace + local welcome page ----------

WORKSPACE_DIR = Path(__file__).with_name("static") / "workspace"     # built by `npm run build` in frontend/
EXTENSION_DIR = Path(__file__).resolve().parent.parent / "extension"
# Explicit allowlist of the files the extension needs (manifest + what it references). Nothing else — no local data,
# alternate manifests, dotfiles or development artefacts — can ever end up in the download.
EXTENSION_FILES = (
    "manifest.json", "background.js", "content.js", "crop.js", "offscreen.html", "offscreen.js",
    "ui-next/popup.html", "ui-next/popup.css", "ui-next/popup.js", "ui-next/logo.svg",
    "ui-next/icons/icon16.png", "ui-next/icons/icon32.png", "ui-next/icons/icon48.png", "ui-next/icons/icon128.png",
)


def _workspace_file(name: str) -> Path:
    path = WORKSPACE_DIR / name
    if not path.is_file():
        raise HTTPException(503, "the MacaroniGraph workspace is not built yet: run `npm ci && npm run build` in frontend/")
    return path


@app.get("/welcome", response_class=HTMLResponse)
def welcome():
    return FileResponse(_workspace_file("welcome.html"), media_type="text/html")


@app.get("/welcome/extension.zip")
def extension_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in EXTENSION_FILES:
            path = EXTENSION_DIR / rel
            if not path.is_file():
                raise HTTPException(500, f"extension file missing: {rel}")
            z.write(path, f"macaronigraph-extension/{rel}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="macaronigraph-extension.zip"'})


@app.get("/workspace")
def workspace_redirect():
    return RedirectResponse("/workspace/")


if WORKSPACE_DIR.is_dir():
    app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_DIR), html=True), name="workspace")


# ---------- start page + permitted test media ----------

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><title>MacaroniGraph — worker tab</title>
<body style="font:16px system-ui;max-width:640px;margin:40px auto">
<h1>MacaroniGraph worker tab</h1>
<p>This tab is the <b>worker tab</b>. Click the extension icon, choose a source, then <b>Start Processing</b>.
The extension captures <i>this</i> tab and navigates it through the queue.</p></body>"""


@app.get("/test-media/page/{rid}", response_class=HTMLResponse)
def test_page(rid: str):
    if not (config.TEST_MEDIA_DIR / f"{rid}.mp4").is_file():
        return HTMLResponse("<!doctype html><title>Page not found</title>"
                            "<h2>Sorry, this page isn't available.</h2>", status_code=404)
    return f"""<!doctype html><title>Test reel {rid}</title>
<body style="margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh">
<video src="/test-media/file/{rid}.mp4" autoplay muted playsinline loop preload="auto"
       style="height:100vh;aspect-ratio:9/16"></video></body>"""


@app.get("/test-media/file/{name}")
def test_file(name: str):
    path = (config.TEST_MEDIA_DIR / name).resolve()
    if path.parent != config.TEST_MEDIA_DIR.resolve() or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")
