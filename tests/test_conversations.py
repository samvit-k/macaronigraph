"""Conversation memory (backend/conversations.py): separate storage, follow-up rewriting rules, referents only from
questions and cited passages (never assistant text), fallbacks."""
import json
import sqlite3

import pytest

from backend import conversations as cv
from backend import jobs, models


@pytest.fixture(autouse=True)
def iso(isolated_output):
    yield


@pytest.fixture
def v1(monkeypatch):
    # The Phase 4B rewrite path (model rewrite with a fallback) stays available as referents="v1"; the default is v2,
    # which never calls the model.
    monkeypatch.setitem(cv.DEFAULTS, "referents", "v1")


def turn(cid, q, q_concepts=(), cited=(), cited_reels=(), answer_text="", status="ok"):
    tid = cv.add_turn(cid, q, standalone_query=q, mode_requested="hybrid", mode_used="hybrid", status=status,
                      question_concepts=[{"key": k.lower(), "name": k} for k in q_concepts],
                      cited_concepts=[{"key": k.lower(), "name": k} for k in cited],
                      cited_reels=list(cited_reels), answer={"text": answer_text})
    return tid


class RewriteLLM:
    def __init__(self, q):
        self.q, self.calls = q, 0

    def generate(self, system, user, max_tokens=0):
        self.calls += 1
        assert models.held_by_me() and models.holder()["purpose"] == "follow-up rewriting"
        return json.dumps({"q": self.q}), {"finish_reason": "stop"}


# ---------------------------------------------------------------- storage

def test_conversations_live_in_their_own_database_file():
    assert cv.db_path() != jobs.db_path() and cv.db_path().name == "conversations.db"
    cid = cv.create("q")
    turn(cid, "What is CUDA?", answer_text="CUDA is NVIDIA's platform UNIQUE-ANSWER-TEXT")
    with jobs.connect() as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        dump = "\n".join(str(tuple(r)) for t in tables for r in con.execute(f"SELECT * FROM {t}"))
    assert not {"conversations", "turns"} & tables and "UNIQUE-ANSWER-TEXT" not in dump


def test_roundtrip_delete_and_expiry():
    cid = cv.create("first question")
    t1 = turn(cid, "a")
    t2 = turn(cid, "b", status="queued")
    c = cv.get(cid)
    assert [t["seq"] for t in c["turns"]] == [1, 2] and cv.get_turn(t1)["question"] == "a"
    assert cv.expire_queued() == 1 and cv.get_turn(t2)["status"] == "expired" and cv.get_turn(t2)["error"]
    assert cv.delete(cid) and cv.get(cid) is None and cv.get_turn(t1) is None
    con = sqlite3.connect(cv.db_path())
    assert con.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0


# ---------------------------------------------------------------- rewriting

def test_single_referent_pronoun_is_substituted_without_the_model():
    cid = cv.create()
    turn(cid, "What does the pastry chef roadmap say about CUDA?", q_concepts=["CUDA"])
    rw = cv.rewrite("What are the prerequisites for learning it?", cv.previous_turns(cid), llm=RewriteLLM("x"))
    assert rw["method"] == "substitution" and rw["standalone"] == "What are the prerequisites for learning CUDA?"


def test_model_rewrite_is_accepted_only_with_question_and_referent_words(v1):
    cid = cv.create()
    turn(cid, "What does my ARM vs x86 Reel say about battery life?", q_concepts=["ARM", "x86"])
    ok = RewriteLLM("How does x86 compare with ARM for gaming?")
    rw = cv.rewrite("How does that compare with the other architecture for gaming?", cv.previous_turns(cid), llm=ok)
    assert rw["method"] == "model" and rw["standalone"] == "How does x86 compare with ARM for gaming?"
    bad = RewriteLLM("How does x86 compare with ARM for gaming on Kubernetes?")
    rw = cv.rewrite("How does that compare with the other architecture for gaming?", cv.previous_turns(cid), llm=bad)
    assert rw["method"] == "fallback" and "kubernetes" in rw["reason"]
    assert rw["standalone"] == "How does that compare with the other architecture for gaming?"
    assert rw["boost"] == ["arm", "x86"]                                           # referents become hints


def test_referents_come_from_questions_and_cited_passages_never_assistant_text():
    cid = cv.create()
    turn(cid, "Tell me about the machine learning security Reel.", q_concepts=[], cited=["Foolbox", "TextAttack"],
         answer_text="It also covers Rust and Kubernetes in depth.")
    names = [r["name"] for r in cv.referents(cv.previous_turns(cid))]
    assert names == ["Foolbox", "TextAttack"] and "Rust" not in names


def test_model_busy_or_capture_active_falls_back_without_failing(v1):
    cid = cv.create()
    turn(cid, "What do my Reels say about ARM and x86?", q_concepts=["ARM", "x86"])
    rw = cv.rewrite("How do they compare?", cv.previous_turns(cid), allow_model=False)
    assert rw["method"] == "fallback" and "capture" in rw["reason"] and rw["boost"] == ["arm", "x86"]

    class Busy:
        def generate(self, *a, **k):
            raise models.HeavyBusy("graph extraction is using the local model")
    rw = cv.rewrite("How do they compare?", cv.previous_turns(cid), llm=Busy())
    assert rw["method"] == "fallback" and "HeavyBusy" in rw["reason"]


def test_non_followups_and_first_turns_are_left_alone():
    assert cv.rewrite("What is CUDA?", [])["method"] == "none"
    cid = cv.create()
    turn(cid, "What is CUDA?", q_concepts=["CUDA"])
    assert cv.rewrite("What is the 3-2-1 method?", cv.previous_turns(cid))["method"] == "none"


def test_other_reels_and_which_of_them_scopes():
    cid = cv.create()
    turn(cid, "Which Reels talk about Claude Code?", q_concepts=["Claude Code"], cited_reels=["A", "B"])
    rw = cv.rewrite("Which other Reels mention this?", cv.previous_turns(cid))
    assert rw["exclude_previous"] and rw["previous_cited_reels"] == ["A", "B"]
    assert rw["standalone"] == "Which other Reels mention Claude Code?"
    rw = cv.rewrite("Which of them shows how to make Wi-Fi faster?", cv.previous_turns(cid))
    assert rw["scope_previous"] and not rw["exclude_previous"]


# ---------------------------------------------------------------- Phase 4C rewriting (v2)

@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setitem(cv.DEFAULTS, "referents", "v2")


def test_v2_pronoun_with_one_entity_named_by_the_user_is_substituted(v2):
    cid = cv.create()
    turn(cid, "What does my Reel say about Firstmate?", q_concepts=["Firstmate"])
    rw = cv.rewrite("Is it free?", cv.previous_turns(cid, 3))
    assert rw["method"] == "substitution" and rw["standalone"] == "Is Firstmate free?"


def test_v2_entity_named_two_turns_back_still_resolves(v2):
    cid = cv.create()
    turn(cid, "What is Firstmate?", q_concepts=["Firstmate"])
    turn(cid, "What did the creator do when the plan ran low?")              # names no entity
    rw = cv.rewrite("Is it free?", cv.previous_turns(cid, 3))
    assert rw["method"] == "substitution" and rw["standalone"] == "Is Firstmate free?"


def test_v2_incidental_cited_resources_never_override_the_subject(v2):
    # Phase 4B pattern: the question named no entity; the cited passage listed resources → "it" became the resources
    cid = cv.create()
    turn(cid, "What does the pastry chef roadmap say I should learn?", q_concepts=[],
         cited=["CUDA Programming Guide", "CUTLASS", "CuTe DSL"])
    rw = cv.rewrite("What are the prerequisites for learning it?", cv.previous_turns(cid, 3), llm=RewriteLLM("x"))
    assert rw["method"] == "context" and "CUTLASS" not in rw["standalone"]
    assert rw["standalone"].startswith("What are the prerequisites for learning it?")
    assert "pastry chef roadmap" in rw["retrieval_query"] and rw["boost"] == []


def test_v2_several_possible_referents_are_not_forced_into_a_substitution(v2, monkeypatch):
    monkeypatch.setitem(cv.DEFAULTS, "clarify", False)          # Phase 4C behaviour; clarify is tested below
    cid = cv.create()
    turn(cid, "What is Firstmate?", q_concepts=["Firstmate"])
    turn(cid, "What happened when Claude ran low and they moved to Codex?", q_concepts=["Claude", "Codex"])
    calls = RewriteLLM("Is Codex free?")
    rw = cv.rewrite("Is it free?", cv.previous_turns(cid, 3), llm=calls)
    assert rw["method"] == "context" and calls.calls == 0 and rw["reason"] == "several possible referents"
    assert "Is it free?" in rw["standalone"] and "What is Firstmate?" in rw["standalone"]
    assert set(rw["boost"]) == {"firstmate", "claude", "codex"}


def test_v2_topic_change_and_self_named_subject_are_left_alone(v2):
    cid = cv.create()
    turn(cid, "What does the inference engineering Reel compare the field to?", q_concepts=["inference engineering"])
    for q in ("Switching topics: what does the Verilog Reel say to learn?",
              "What about the Platonic Representation Hypothesis Reel — what does it claim?"):
        rw = cv.rewrite(q, cv.previous_turns(cid, 3))
        assert rw["method"] == "none" and rw["standalone"] == q and not rw["boost"]


def test_v2_standalone_questions_and_other_reels_without_a_reference_are_unchanged(v2):
    cid = cv.create()
    turn(cid, "Which Reel talks about OpenAI hiring?", q_concepts=["OpenAI"], cited_reels=["A"])
    rw = cv.rewrite("Which other Reels mention LeetCode?", cv.previous_turns(cid, 3))
    assert rw["method"] == "none" and rw["standalone"] == "Which other Reels mention LeetCode?"
    assert rw["exclude_previous"] and rw["previous_cited_reels"] == ["A"]
    assert cv.rewrite("What is the 3-2-1 method?", cv.previous_turns(cid, 3))["method"] == "none"


def test_v2_which_of_them_is_scoping_not_a_pronoun_to_resolve(v2):
    cid = cv.create()
    turn(cid, "Which Reels talk about Claude Code?", q_concepts=["Claude Code"], cited_reels=["A", "B"])
    rw = cv.rewrite("Which of them shows how to make Wi-Fi faster?", cv.previous_turns(cid, 3))
    assert rw["scope_previous"] and rw["method"] == "none"


def test_v2_history_is_isolated_between_conversations(v2):
    a, b = cv.create(), cv.create()
    turn(a, "What is Firstmate?", q_concepts=["Firstmate"])
    turn(b, "What is CUDA?", q_concepts=["CUDA"])
    assert cv.rewrite("Is it free?", cv.previous_turns(a, 3))["standalone"] == "Is Firstmate free?"
    assert cv.rewrite("Is it free?", cv.previous_turns(b, 3))["standalone"] == "Is CUDA free?"


def test_v2_never_uses_assistant_text_as_context(v2):
    cid = cv.create()
    turn(cid, "What does the roadmap say?", answer_text="It recommends Rust and Kubernetes UNIQUE-ANSWER")
    rw = cv.rewrite("What are the prerequisites for it?", cv.previous_turns(cid, 3))
    assert "UNIQUE-ANSWER" not in rw["standalone"] + rw.get("retrieval_query", "")


# ---------------------------------------------------------------- Phase 4D disambiguation (clarify)

@pytest.fixture
def clar(monkeypatch):
    monkeypatch.setitem(cv.DEFAULTS, "referents", "v2")
    monkeypatch.setitem(cv.DEFAULTS, "clarify", True)


def test_singular_pronoun_after_two_named_things_asks_instead_of_guessing(clar):
    cid = cv.create()
    turn(cid, "What are Quillmark and Paperhawk used for?", q_concepts=["Quillmark", "Paperhawk"])
    rw = cv.rewrite("Is it open source?", cv.previous_turns(cid, 3))
    assert rw["method"] == "clarify" and rw["clarify"] == ["Quillmark", "Paperhawk"]
    assert rw["standalone"] == "Is it open source?"


def test_clarify_is_off_by_default_and_plural_or_single_references_are_unchanged(monkeypatch, clar):
    cid = cv.create()
    turn(cid, "What are Quillmark and Paperhawk used for?", q_concepts=["Quillmark", "Paperhawk"])
    assert cv.rewrite("Are they free?", cv.previous_turns(cid, 3))["method"] == "context"       # plural: both
    one = cv.create()
    turn(one, "What does my Reel say about Firstmate?", q_concepts=["Firstmate"])
    assert cv.rewrite("Is it free?", cv.previous_turns(one, 3))["standalone"] == "Is Firstmate free?"
    monkeypatch.setitem(cv.DEFAULTS, "clarify", False)
    assert cv.rewrite("Is it open source?", cv.previous_turns(cid, 3))["method"] == "context"


@pytest.mark.parametrize("q", ["How does that compare with the other architecture for gaming?",
                               "What is the difference between them for gaming?", "Is it better for both?"])
def test_comparison_wording_refers_to_both_things_and_is_not_clarified(clar, q):
    cid = cv.create()
    turn(cid, "What does my ARM vs x86 Reel say about battery life?", q_concepts=["ARM", "x86"])
    assert cv.rewrite(q, cv.previous_turns(cid, 3))["method"] == "context"


def test_an_entity_from_an_earlier_turn_is_not_a_reason_to_clarify(clar):
    cid = cv.create()
    turn(cid, "What is Firstmate?", q_concepts=["Firstmate"])
    turn(cid, "What does the creator use for speech to text?", q_concepts=["speech to text"])
    rw = cv.rewrite("Is it free?", cv.previous_turns(cid, 3))
    assert rw["method"] != "clarify"


def test_that_step_keeps_the_earlier_questions_as_context(clar):
    cid = cv.create()
    turn(cid, "What does the electronics roadmap say step 2 is?", q_concepts=["electronics"])
    rw = cv.rewrite("What resource does it recommend for that step?", cv.previous_turns(cid, 3))
    assert rw["method"] == "context" and "step 2" in rw["retrieval_query"]


def test_delete_all_removes_every_conversation_and_turn_only():
    a, b = cv.create(), cv.create()
    turn(a, "What is CUDA?")
    turn(b, "What is ARM?")
    assert cv.delete_all() == 2
    assert cv.list_conversations() == [] and cv.get(a) is None and cv.get(b) is None
    assert cv.delete_all() == 0
