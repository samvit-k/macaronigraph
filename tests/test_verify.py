"""Phase 4D claim verification, bounded repair and multi-part coverage (backend/verify.py) with a scripted model:
deterministic checks, decision rule, drop/repair/flag modes, re-verification of repairs, the call budget, unreadable
verifier output, and that one answer's model calls all run inside one hold of the heavy lock."""
import json

import pytest

from backend import answer, models, verify


@pytest.fixture(autouse=True)
def iso(isolated_output, monkeypatch):
    monkeypatch.setitem(answer.DEFAULTS, "inline_citations", True)
    yield


def src(sid, rid, text, kind="transcript"):
    return {"sid": sid, "chunk_id": f"{rid}:{kind}:000", "reel_id": rid, "source_url": f"https://x/reel/{rid}/",
            "owner": "c", "content_type": kind, "start": 0.0, "end": 5.0, "text": text}


PACK = {"sources": [src("S1", "A", "Round 2 is hands-on service design. They skip puzzle questions entirely."),
                    src("S2", "B", "A GPU like the Zephyr X9 has around 18,000 small cores."),
                    src("S3", "C", "SALE 50% OFF", kind="ocr")], "connections": {}}


class Script:
    """Returns the next scripted reply for each kind of call; records calls and checks the heavy lock."""

    def __init__(self, draft, verify_replies=(), repair=None):
        self.draft, self.ver, self.repair, self.calls = draft, list(verify_replies), repair, []

    def generate(self, system, user, max_tokens=0):
        assert models.held_by_me()
        if system == verify.VERIFY_SYSTEM:
            self.calls.append("verify")
            r = self.ver.pop(0) if self.ver else ""
        elif system == verify.REPAIR_SYSTEM:
            self.calls.append("repair")
            r = self.repair or ""
        else:
            self.calls.append("draft")
            r = self.draft
        return (r if isinstance(r, str) else json.dumps(r)), {"finish_reason": "stop"}


def v(*labels, quote=""):
    return {"v": [{"id": f"C{i + 1}", "label": lab, "q": quote} for i, lab in enumerate(labels)]}


DRAFT = {"s": [{"t": "The Reel says round 2 is hands-on service design.", "c": ["S1"], "k": "src"},
               {"t": "The Reel says the Zephyr X9 has 25,000 cores.", "c": ["S2"], "k": "src"}]}


def run(mode, llm, coverage=False, q="What happens in round 2, and how many cores does the Zephyr X9 have?"):
    import copy
    answer.DEFAULTS.update(verify=mode, coverage=coverage)
    try:
        return answer.respond(q, copy.deepcopy(PACK), llm=llm, timeout=5)
    finally:
        answer.DEFAULTS.update(verify="off", coverage=False)


# ---------------------------------------------------------------- deterministic checks

def test_numbers_are_normalised_and_versions_handles_and_ids_ignored():
    assert verify.numbers("$10K a month, 33 a day, four parts, 1.137 Hartree, 2.1x") == {10000.0, 33.0, 4.0, 1.137, 2.1}
    assert verify.numbers("ten thousand") == {10000.0}
    assert verify.numbers("Handy v0.10.1, GPT-4.1, @3x.demo.studio, cites S3, one of them") == set()


def test_checks_flag_missing_numbers_hard_and_other_hints_soft():
    p = [PACK["sources"][1]]
    assert verify.checks("It has 18,000 cores.", p, "q")["hard"] == []
    assert "number not in the cited passages: 25000" in verify.checks("It has 25,000 cores.", p, "q")["hard"][0]
    assert verify.checks("Round 2 is coding.", [PACK["sources"][0]], "What happens in round 2?")["hard"] == []
    soft = verify.checks("The GPU is not faster than a CPU.", p, "q")["soft"]
    assert any(s.startswith("comparison") for s in soft) and any(s.startswith("negation") for s in soft)
    assert any("OCR" in s for s in verify.checks("Everything is half price.", [PACK["sources"][2]], "q")["soft"])


@pytest.mark.parametrize("q,n", [
    ("Which companies offer fellowships, and when are their deadlines?", 2),
    ("How much does the course cost, and how long does it take to finish?", 2),
    ("What are the four main things every job needs?", 1),
    ("How does a club compare with personal projects?", 1),
    ("What does the Reel say about sourdough starters?", 0),
])
def test_only_multi_part_questions_are_decomposed(q, n):
    assert len(verify.parts(q)) == n


# ---------------------------------------------------------------- modes

def test_drop_removes_problem_claims_and_keeps_supported_ones_unchanged():
    llm = Script(DRAFT, [v("SUPPORTED", "SUPPORTED")])
    out = run("drop", llm)
    assert [s["text"] for s in out["sentences"]] == ["The Reel says round 2 is hands-on service design."]
    assert out["removed"][-1]["reason"].startswith("unverified: number not in the cited passages")
    assert llm.calls == ["draft", "verify"] and out["verification"]["calls"] == 1


def test_contradicted_or_not_found_claims_are_problems_but_inferred_is_kept():
    out = run("drop", Script(DRAFT | {"s": DRAFT["s"][:1]}, [v("INFERRED")]))
    assert out["status"] == "ok"
    out = run("drop", Script(DRAFT | {"s": DRAFT["s"][:1]}, [v("NOT_FOUND")]))
    assert out["status"] == "insufficient_evidence" and out["text"] == answer.ABSTAIN


def test_repair_fixes_a_claim_and_the_fix_is_reverified_before_delivery():
    fix = {"r": [{"id": "C1", "t": "The Reel says the Zephyr X9 has around 18,000 small cores.", "c": ["S2"]}]}
    llm = Script(DRAFT, [v("SUPPORTED", "SUPPORTED"), v("SUPPORTED")], repair=fix)
    out = run("repair", llm)
    assert [s["text"] for s in out["sentences"]] == ["The Reel says round 2 is hands-on service design.",
                                                     "The Reel says the Zephyr X9 has around 18,000 small cores."]
    assert out["sentences"][1]["repaired"] and llm.calls == ["draft", "verify", "repair", "verify"]
    assert out["verification"]["calls"] == 3 and out["removed"][-1]["reason"].startswith("repaired:")


def test_a_repair_that_fails_reverification_or_validation_is_never_delivered():
    bad = {"r": [{"id": "C1", "t": "The Reel says it has 18,000 cores.", "c": ["S2"]}]}
    out = run("repair", Script(DRAFT, [v("SUPPORTED", "SUPPORTED"), v("NOT_FOUND")], repair=bad))
    assert len(out["sentences"]) == 1
    out = run("repair", Script(DRAFT, [v("SUPPORTED", "SUPPORTED"), ""], repair=bad))          # no verdict
    assert len(out["sentences"]) == 1
    wrong = {"r": [{"id": "C1", "t": "The Reel says it has 18,000 cores.", "c": ["S9"]}]}       # invalid citation
    llm = Script(DRAFT, [v("SUPPORTED", "SUPPORTED")], repair=wrong)
    out = run("repair", llm)
    assert len(out["sentences"]) == 1 and llm.calls == ["draft", "verify", "repair"]        # nothing to re-verify


def test_unreadable_verifier_output_keeps_original_claims_marked_unverified():
    out = run("drop", Script(DRAFT | {"s": DRAFT["s"][:1]}, ["not json"]))
    assert out["status"] == "ok" and out["verification"]["claims"][0]["decision"] == "unverified"


def test_flag_mode_annotates_without_changing_the_answer():
    out = run("flag", Script(DRAFT, [v("SUPPORTED", "NOT_FOUND")]))
    assert len(out["sentences"]) == 2 and out["verification"]["claims"][1]["decision"] == "problem"


def test_coverage_missing_part_is_answered_by_repair_or_noted():
    one = {"s": DRAFT["s"][:1]}
    ver = {"v": [{"id": "C1", "label": "SUPPORTED", "q": ""}], "parts": [{"id": "P1", "answered": True},
                                                                         {"id": "P2", "answered": False}]}
    add = {"r": [], "add": [{"t": "The Reel says the Zephyr X9 has around 18,000 small cores.", "c": ["S2"]}]}
    llm = Script(one, [ver, v("SUPPORTED")], repair=add)
    out = run("repair", llm, coverage=True)
    assert len(out["sentences"]) == 2 and out["missing"] == "" and out["verification"]["missing_parts"]
    out = run("drop", Script(one, [ver]), coverage=True)
    assert len(out["sentences"]) == 1 and out["missing"] == answer.PART_MISSING


def test_call_budget_is_at_most_four_model_calls_per_answer():
    fix = {"r": [{"id": "C1", "t": "The Reel says the Zephyr X9 has around 18,000 small cores.", "c": ["S2"]}]}
    llm = Script(DRAFT, [v("NOT_FOUND", "NOT_FOUND"), v("NOT_FOUND")], repair=fix)
    run("repair", llm, coverage=True)
    assert len(llm.calls) <= 4


def test_off_by_default_means_a_single_generation_call():
    llm = Script(DRAFT)
    answer.respond("q", {"sources": list(PACK["sources"]), "connections": {}}, llm=llm, timeout=5)
    assert llm.calls == ["draft"]
