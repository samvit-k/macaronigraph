"""FastAPI backend: capture uploads, job status, search, knowledge graph, answers, workspace and welcome page.

Run:  .venv/bin/uvicorn backend.app:app --port 8765     (uvicorn binds 127.0.0.1 by default; keep it local)
"""
from __future__ import annotations

import io
import json
import sys
import uuid
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, evalframes, extension_package, indexer, jobs, metadata, worker

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
        indexer.start()
        _graph_call("start")
        _answers_call("start")
    yield
    if config.ASYNC_PROCESSING:
        _answers_call("stop")
        _graph_call("stop")
        indexer.stop()
    worker.stop()
    try:
        from . import models
        models.shutdown()
    except Exception as e:  # noqa: BLE001
        print(f"[models] shutdown failed: {type(e).__name__}: {e}", file=sys.stderr)

def _answers_call(fn: str):
    try:
        from . import ask
        return getattr(ask, fn)()
    except Exception as e:  # noqa: BLE001
        print(f"[answers] {fn} failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None

def _graph_call(fn: str):
    try:
        from .graph import sync as graph_sync
        return getattr(graph_sync, fn)()
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}

@app.get("/api/session")
def session(source: str = "test", limit: int = 3, offset: int = 0, mode: str = "pending"):
    if source not in SOURCES:
        raise HTTPException(400, f"source must be one of {list(SOURCES)}")
    if mode not in ("pending", "all", "eval"):
        raise HTTPException(400, "mode must be 'pending', 'all' or 'eval'")
    jobs.import_posts(source, list(_posts(source).values()))
    sid = f"{source}-{uuid.uuid4().hex[:8]}"
    ids = evalframes.selection() if mode == "eval" else None
    out = {"session_id": sid, "source": source, "mode": mode,
           "items": jobs.create_session(sid, source, limit, offset, mode, reel_ids=ids)}
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
    return {**jobs.session_progress(session_id), "backlog": len(worker.pending_jobs())}

@app.get("/api/status")
def job_status(source: str | None = None):
    return {**jobs.status(source), "backlog": len(worker.pending_jobs()), "quarantined": len(worker.dead_jobs())}

@app.post("/api/process")
def process(meta: str = Form(...), audio: UploadFile | None = File(None),
            frames: list[UploadFile] = File(default_factory=list)):
    try:
        cap = json.loads(meta)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"meta is not JSON: {e}")
    uploads = [u for u in [audio, *frames] if u is not None]
    try:
        cap.setdefault("session_id", f"nosession-{uuid.uuid4().hex[:8]}")
        sid, rid = cap.get("session_id"), cap.get("reel_id")
        prior = jobs.existing_attempt(sid, rid)
        if prior:
            return {"reel_id": rid, "status": prior["status"], "errors": json.loads(prior["errors"]),
                    "duplicate": True, "attempt_id": prior["attempt_id"]}
        if worker.find_job(sid, rid):
            return {"reel_id": rid, "status": "queued", "errors": [], "duplicate": True, "queued": True}
        if config.ASYNC_PROCESSING:
            try:
                backlog = worker.ensure_capacity()
            except worker.Overloaded as e:
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
    out = worker.process_job(job, _lookup)
    if out is None:
        raise HTTPException(500, "capture spooled but could not be persisted; it will be retried")
    return out

@app.get("/api/results")
def results(session_id: str | None = None):
    with jobs.connect() as con:
        q = "SELECT attempt_id, reel_id, session_id, processed_at, status, outcome, errors FROM attempts"
        rows = con.execute(q + (" WHERE session_id=?" if session_id else "") + " ORDER BY attempt_id",
                           (session_id,) if session_id else ()).fetchall()
    return [{**dict(r), "errors": json.loads(r["errors"] or "[]")} for r in rows]

@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=1000), k: int = Query(10, ge=1, le=50),
           source: str = "instagram", types: str | None = None, group: str = "reel"):
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
    try:
        return indexer.request_sync(rebuild=rebuild, wait=wait)
    except indexer.WriterBusy as e:
        raise HTTPException(409, str(e))

@contextmanager
def _graph_store():
    from .graph.store import GraphOffline, Neo4jStore
    store = Neo4jStore()
    try:
        yield store
    except GraphOffline as e:
        raise HTTPException(503, f"knowledge graph offline: {e}")
    finally:
        store.close()

def _graph(fn, *args, **kw):
    from .graph import queries as gq
    with _graph_store() as store:
        return getattr(gq, fn)(store, *args, **kw)

@app.get("/api/graph/status")
def graph_status():
    from .graph import queries as gq
    return gq.status()

@app.post("/api/graph/sync")
def graph_sync(force: bool = False, wait: bool = False):
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

def _view(fn, *args, **kw):
    from .graph import view as gv
    try:
        with _graph_store() as store:
            return getattr(gv, fn)(store, *args, **kw)
    except ValueError as e:
        raise HTTPException(400, str(e))

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

class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None
    mode: str = "hybrid"
    generate: bool = True
    options: dict | None = None

_ASK_OPTIONS = {"co_mention", "relations", "max_sources"}

@app.post("/api/ask")
def ask_endpoint(body: AskBody):
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

WORKSPACE_DIR = Path(__file__).with_name("static") / "workspace"

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
    try:
        extension_package.write_zip(buf)
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="macaronigraph-extension.zip"'})

@app.get("/workspace")
def workspace_redirect():
    return RedirectResponse("/workspace/")

if WORKSPACE_DIR.is_dir():
    app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_DIR), html=True), name="workspace")

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