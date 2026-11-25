"""POST /api/ask and the request flow (backend/ask.py) — integration over a real temporary Chroma index, a fake
graph or an offline Neo4j, and a scripted local model: fallback, deferral and polling, queue capacity, worker
recovery, restart expiry, previous-Reel scoping, cited concepts, standalone queries, and the /app page."""
import json
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend import ask, config, conversations, embedder, indexer, jobs, models
from backend.app import app
from test_indexer import add
from test_rag_retrieve import FakeGraph

client = TestClient(app)          # no lifespan: background threads are not started


class AnswerLLM:
    """Cites the first source with its first words; records prompts; answers rewrite requests if configured."""

    def __init__(self, rewrite_to=None):
        self.prompts, self.rewrite_to = [], rewrite_to

    def generate(self, system, user, max_tokens=0):
        assert models.held_by_me()
        self.prompts.append(user)
        if "standalone question" in system:
            return json.dumps({"q": self.rewrite_to or ""}), {"finish_reason": "stop"}
        sid, text = re.search(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', user, re.S).groups()
        return json.dumps({"s": [{"t": "The Reel says " + " ".join(text.split()[:6]) + ".", "c": [sid],
                                  "k": "src"}]}), {"finish_reason": "stop"}


@pytest.fixture(autouse=True)
def env(isolated_output, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", embedder.HashEmbedder.MODEL)
    monkeypatch.setattr(embedder, "_cache", {})
    monkeypatch.setattr(indexer, "_clients", {})
    monkeypatch.setattr(ask, "_queue", __import__("collections").deque())
    monkeypatch.setattr(config, "RAG_HEAVY_WAIT_S", 0.5)
    jobs._export_state.update(pending=0, last=0.0)
    llm = AnswerLLM()
    monkeypatch.setattr(models, "shared_llm", lambda: llm)
    add("A", "CUDA kernels run on thousands of GPU cores")
    add("B", "Triton is built on top of CUDA for kernels")
    add("C", "sourdough bread starter recipe")
    indexer.sync_all()
    yield llm
    indexer.release_writer()


def capture_on():
    jobs.create_session("instagram-live", "instagram", 0)


def capture_off():
    with jobs.tx() as con:
        con.execute("UPDATE sessions SET finished_at = 1")


# ---------------------------------------------------------------- API basics

def test_hybrid_with_neo4j_offline_falls_back_to_vector_and_still_answers(env):
    r = client.post("/api/ask", json={"question": "How do CUDA kernels work?"}).json()
    assert r["mode_requested"] == "hybrid" and r["mode_used"] == "vector"
    assert r["fallback"] and r["fallback"][0]["to"] == "vector" and "graph unavailable" in r["fallback"][0]["reason"]
    assert r["status"] == "ok" and r["answer"]["sentences"] and r["citations"]
    cited = {s["sid"]: s for s in r["sources"]}
    for c in r["citations"]:
        assert c["sid"] in cited and c["chunk_id"] == cited[c["sid"]]["chunk_id"] and c["source_url"]
    assert set(r["timing"]) == {"retrieval", "rewrite_s", "generation"} and r["timing"]["generation"]["generate_s"] >= 0


def test_retrieval_only_when_generation_is_not_requested(env):
    r = client.post("/api/ask", json={"question": "How do CUDA kernels work?", "generate": False}).json()
    assert r["status"] == "retrieval_only" and r["sources"] and r["answer"]["status"] == "unavailable"
    assert env.prompts == []


@pytest.mark.parametrize("body,code", [({"question": "x", "mode": "magic"}, 400), ({"question": ""}, 422),
                                       ({"question": "x", "conversation_id": "nope"}, 400)])
def test_bad_requests(body, code):
    assert client.post("/api/ask", json=body).status_code == code


def test_unknown_request_and_conversation_are_404():
    assert client.get("/api/ask/999999").status_code == 404
    assert client.get("/api/conversations/nope").status_code == 404
    assert client.delete("/api/conversations/nope").status_code == 404


def test_no_index_is_503(monkeypatch):
    monkeypatch.setattr(indexer, "active_collection", lambda: None)
    assert client.post("/api/ask", json={"question": "CUDA?"}).status_code == 503


def test_legacy_app_urls_redirect_to_the_workspace():
    for path in ("/app", "/app/", "/app/chat.js"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 307 and r.headers["location"] == "/workspace/"


def test_conversation_endpoints_roundtrip(env):
    r = client.post("/api/ask", json={"question": "How do CUDA kernels work?"}).json()
    c = client.get(f"/api/conversations/{r['conversation_id']}").json()
    assert [t["question"] for t in c["turns"]] == ["How do CUDA kernels work?"]
    assert any(x["id"] == r["conversation_id"] for x in client.get("/api/conversations").json())
    assert client.delete(f"/api/conversations/{r['conversation_id']}").status_code == 200


# ---------------------------------------------------------------- deferral, queue, worker

def test_capture_defers_generation_then_the_queue_answers_with_the_standalone_query(env, monkeypatch):
    monkeypatch.setitem(ask.answer.DEFAULTS, "listing", False)   # this test is about the generative path
    first = ask.ask("What is CUDA?", graph=FakeGraph(), llm=env)
    capture_on()
    r = ask.ask("Which other Reels mention it?", conversation_id=first["conversation_id"], graph=FakeGraph())
    assert r["status"] == "queued" and r["answer"]["status"] == "deferred" and r["sources"]
    assert "capture" in r["answer"]["reason"]
    assert client.get(f"/api/ask/{r['request_id']}").json()["status"] == "queued"
    assert ask.process_one(llm=env) is None                       # rechecked right before generating: still busy
    capture_off()
    n = len(env.prompts)
    assert ask.process_one(llm=env) == r["request_id"]
    done = client.get(f"/api/ask/{r['request_id']}").json()
    assert done["status"] == "ok" and done["question"] == "Which other Reels mention it?"
    assert done["standalone_query"] == "Which other Reels mention CUDA?"
    assert "QUESTION: Which other Reels mention CUDA?" in env.prompts[n]  # generated for the standalone query


def test_model_busy_beyond_the_wait_defers_instead_of_failing(env):
    entered, release = threading.Event(), threading.Event()

    def hold():
        with models.heavy("graph extraction"):
            entered.set()
            release.wait(10)
    t = threading.Thread(target=hold)
    t.start()
    entered.wait(5)
    try:
        r = ask.ask("How do CUDA kernels work?", llm=env)
        assert r["status"] == "queued" and "graph extraction" in r["answer"]["reason"]
    finally:
        release.set()
        t.join(5)
    assert ask.process_one(llm=env) == r["request_id"] and ask.request_status(r["request_id"])["status"] == "ok"


def test_queue_capacity_is_reported_not_silently_dropped(env, monkeypatch):
    monkeypatch.setattr(ask, "QUEUE_MAX", 2)
    capture_on()
    rs = [ask.ask(f"How do CUDA kernels work, {i}?") for i in range(3)]
    assert [r["status"] for r in rs] == ["queued", "queued", "unavailable"]
    assert "queue full" in rs[2]["answer"]["reason"] and rs[2]["sources"]
    assert ask.queued() == [rs[0]["request_id"], rs[1]["request_id"]]


def test_worker_recovers_from_a_failing_item_and_keeps_going(env, monkeypatch):
    capture_on()
    bad = ask.ask("How do CUDA kernels work?")
    good = ask.ask("Tell me about Triton kernels")
    capture_off()
    real = conversations.get_turn
    calls = {"n": 0}

    def flaky(tid):
        calls["n"] += 1
        if tid == bad["request_id"] and calls["n"] == 1:
            raise RuntimeError("database hiccup")
        return real(tid)
    monkeypatch.setattr(conversations, "get_turn", flaky)
    monkeypatch.setattr(ask, "POLL_S", 0.05)
    ask._stop.clear()
    worker = threading.Thread(target=ask._loop)                  # the worker loop itself (no restart expiry)
    worker.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and real(good["request_id"])["status"] == "queued":
            ask._wake.set()
            time.sleep(0.05)
    finally:
        ask._stop.set()
        ask._wake.set()
        worker.join(10)
    assert real(bad["request_id"])["status"] == "error" and "database hiccup" in real(bad["request_id"])["error"]
    assert real(good["request_id"])["status"] == "ok"


def test_restart_expires_queued_turns_visibly(env):
    capture_on()
    r = ask.ask("How do CUDA kernels work?")
    ask._queue.clear()                                           # the process "restarts"
    assert ask.start() == 1
    ask.stop()
    t = ask.request_status(r["request_id"])
    assert t["status"] == "expired" and "restarted" in t["answer"]["reason"]


# ---------------------------------------------------------------- scoping and memory

def test_other_reels_exclude_previously_cited_reels_and_which_of_them_restricts(env, monkeypatch):
    monkeypatch.setitem(ask.answer.DEFAULTS, "listing", False)   # this test is about the generative path
    first = ask.ask("Which Reels mention CUDA?", graph=FakeGraph(), llm=env)
    cited = set(conversations.get_turn(first["turn_id"])["cited_reels"])
    assert cited
    other = ask.ask("Which other Reels mention this?", conversation_id=first["conversation_id"],
                    graph=FakeGraph(), llm=env)
    assert not cited & set(other["connections"]["mentions"][0]["reels"])
    assert not cited & {s["reel_id"] for s in other["sources"]}
    previous = set(conversations.get_turn(other["turn_id"])["cited_reels"])    # "them" = the last answer's Reels
    assert previous
    scoped = ask.ask("Which of them talk about Triton?", conversation_id=first["conversation_id"],
                     graph=FakeGraph(), llm=env)
    assert scoped["sources"] and {s["reel_id"] for s in scoped["sources"]} <= previous


def test_cited_concepts_come_from_the_exact_cited_chunks(env):
    r = ask.ask("Triton is built on top of CUDA", graph=FakeGraph(), llm=env)
    t = conversations.get_turn(r["turn_id"])
    cited_chunks = {c["chunk_id"] for c in r["citations"]}
    expected = {c["key"] for ch in cited_chunks for c in t["retrieval"]["source_concepts"].get(ch, [])}
    assert expected and {c["key"] for c in t["cited_concepts"]} == expected
    assert t["question_concepts"] and all(c["name"] for c in t["question_concepts"])


def test_assistant_answers_never_reach_knowledge_stores(env):
    ask.ask("How do CUDA kernels work?", llm=env)
    with jobs.connect() as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert con.execute("SELECT COUNT(*) FROM graph_extraction").fetchone()[0] == 0 if "graph_extraction" in tables else True
        dump = "\n".join(str(tuple(r)) for tb in tables for r in con.execute(f"SELECT * FROM {tb}"))
    assert "The Reel says" not in dump
    col = indexer.get_collection(indexer.active_collection())
    assert not any("The Reel says" in d for d in col.get(include=["documents"])["documents"])


def test_v2_context_rewrite_drives_retrieval_and_generation_end_to_end(env, monkeypatch):
    monkeypatch.setitem(conversations.DEFAULTS, "referents", "v2")
    first = ask.ask("How do Triton and CUDA kernels relate?", graph=FakeGraph(), llm=env)
    r = ask.ask("How are they different?", conversation_id=first["conversation_id"], graph=FakeGraph(), llm=env)
    t = conversations.get_turn(r["turn_id"])
    assert t["rewrite"]["method"] == "context"
    assert t["rewrite"]["retrieval_query"] == "How are they different? How do Triton and CUDA kernels relate?"
    assert "Earlier questions in this conversation" in env.prompts[-1] and "The Reel says" not in t["rewrite"]["retrieval_query"]


def test_follow_up_turns_are_marked_so_hard_gates_skip_them(env, monkeypatch):
    seen = []
    real = ask.answer.generate
    monkeypatch.setattr(ask.answer, "generate", lambda *a, **k: seen.append(k.get("follow_up")) or real(*a, **k))
    monkeypatch.setitem(conversations.DEFAULTS, "referents", "v2")
    first = ask.ask("How do Triton and CUDA kernels relate?", graph=FakeGraph(), llm=env)
    ask.ask("How are they different?", conversation_id=first["conversation_id"], graph=FakeGraph(), llm=env)
    assert seen == [False, True]


def test_which_of_them_follow_ups_skip_hard_gates_only_with_the_fix(env, monkeypatch):
    """Phase 4D fix (held-out hm1.t2): 'Which of those …' strips to a question with nothing to resolve, so the
    rewrite method stays 'none' — but the turn is scoped to the previous answer, i.e. context-dependent."""
    seen = []
    real = ask.answer.generate
    monkeypatch.setattr(ask.answer, "generate", lambda *a, **k: seen.append(k.get("follow_up")) or real(*a, **k))
    monkeypatch.setitem(conversations.DEFAULTS, "referents", "v2")
    monkeypatch.setitem(ask.answer.DEFAULTS, "listing", False)   # this test is about the generative path
    for fix in (False, True):
        monkeypatch.setitem(ask.answer.DEFAULTS, "scoped_follow_up", fix)
        first = ask.ask("Which Reels mention CUDA?", graph=FakeGraph(), llm=env)
        r = ask.ask("Which of those talk about Triton?", conversation_id=first["conversation_id"], graph=FakeGraph(), llm=env)
        assert conversations.get_turn(r["turn_id"])["rewrite"]["method"] == "none"
    assert seen == [False, False, False, True]


def test_ambiguous_follow_up_gets_a_clarifying_question_without_the_model(env, monkeypatch):
    monkeypatch.setitem(conversations.DEFAULTS, "referents", "v2")
    monkeypatch.setitem(conversations.DEFAULTS, "clarify", True)
    first = ask.ask("How do Triton and CUDA kernels relate?", graph=FakeGraph(), llm=env)
    n = len(env.prompts)
    r = ask.ask("Is it free?", conversation_id=first["conversation_id"], graph=FakeGraph(), llm=env)
    assert r["status"] == "clarify" and "Which one do you mean" in r["answer"]["text"] and len(env.prompts) == n
