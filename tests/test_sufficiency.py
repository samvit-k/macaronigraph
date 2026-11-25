"""Evidence-sufficiency signals and their use in answer generation."""
import pytest

from backend import answer, sufficiency as sf
from test_answer import ask, src

@pytest.fixture(autouse=True)
def iso(isolated_output):
    yield

def pack(sim, text="ARM grew from RISC, short fixed commands."):
    return {"sources": [{**src("S1", "A", text), "similarity": sim}], "connections": {}}

def test_signals_report_best_similarity_and_missing_question_terms():
    sig = sf.signals("How does ARM compare with quantum annealing?", pack(0.61))
    assert sig["best_sim"] == 0.61 and "quant" in sig["missing"] and "arm" not in sig["missing"]
    assert 0 < sig["coverage"] < 1

def test_graph_only_passages_have_no_similarity_and_never_trigger_the_similarity_gate():
    sig = sf.signals("ARM?", pack(None))
    assert sig["best_sim"] is None and sf.decide("hard_sim", sig) is None

def test_off_means_no_gate_and_generation_happens(monkeypatch):
    monkeypatch.setitem(answer.DEFAULTS, "sufficiency", "off")
    out, llm = ask({"s": [{"t": "ARM grew from RISC.", "c": ["S1"], "k": "src"}]}, pack=pack(0.1))
    assert llm.calls == 1 and out["status"] == "ok" and out["gate"] is None

def test_hard_similarity_gate_abstains_without_generating(monkeypatch):
    monkeypatch.setitem(answer.DEFAULTS, "sufficiency", "hard_sim")
    out, llm = ask({"s": [{"t": "ARM grew from RISC.", "c": ["S1"], "k": "src"}]}, pack=pack(0.3))
    assert llm.calls == 0 and out["status"] == "insufficient_evidence" and out["gate"]["abstained"]
    assert out["text"] == answer.ABSTAIN

def test_hard_similarity_gate_lets_a_well_matched_question_through(monkeypatch):
    monkeypatch.setitem(answer.DEFAULTS, "sufficiency", "hard_sim")
    out, llm = ask({"s": [{"t": "ARM grew from RISC.", "c": ["S1"], "k": "src"}]}, pack=pack(0.8))
    assert llm.calls == 1 and out["status"] == "ok" and not out["gate"]["abstained"]

def test_soft_mode_warns_the_model_about_missing_terms_but_never_decides_itself(monkeypatch):
    monkeypatch.setitem(answer.DEFAULTS, "sufficiency", "soft")
    out, llm = ask({"s": [{"t": "ARM grew from RISC.", "c": ["S1"], "k": "src"}]},
                   q="How does ARM compare with quantum annealing?", pack=pack(0.2))
    note = llm.user.split("occur in none of the sources: ", 1)[1].split(".", 1)[0]
    assert llm.calls == 1 and "quantum" in note and "annealing" in note and "arm" not in note.split(", ")
    assert out["status"] == "ok"

def test_a_supported_negative_answer_is_kept_as_an_answer(monkeypatch):
    monkeypatch.setitem(answer.DEFAULTS, "sufficiency", "soft")
    p = pack(0.8, "You do not need to master shader math to become a game designer.")
    out, _ = ask({"s": [{"t": "The Reel says you do not need to master shader math.", "c": ["S1"], "k": "src"}]},
                 q="Do I need shader math to become a game designer?", pack=p)
    assert out["status"] == "ok"

def test_the_gate_does_not_bypass_citation_validation(monkeypatch):
    monkeypatch.setitem(answer.DEFAULTS, "sufficiency", "hard_sim")
    out, _ = ask({"s": [{"t": "The answer is 42.", "c": ["S9"], "k": "src"}]}, pack=pack(0.9))
    assert out["status"] == "insufficient_evidence"

def test_hard_sim_anchor_needs_both_low_similarity_and_a_missing_term():
    low_missing = {"best_sim": 0.40, "missing": ["tire"], "missing_words": ["tire"], "coverage": 0.5}
    low_covered = {"best_sim": 0.45, "missing": [], "missing_words": [], "coverage": 1.0}
    high_missing = {"best_sim": 0.70, "missing": ["lru"], "missing_words": ["lru"], "coverage": 0.75}
    assert sf.decide("hard_sim_anchor", low_missing, 0.55) and "tire" in sf.decide(
        "hard_sim_anchor", low_missing, 0.55)
    assert sf.decide("hard_sim_anchor", low_covered, 0.55) is None
    assert sf.decide("hard_sim_anchor", high_missing, 0.55) is None
    assert sf.decide("hard_sim_anchor", {**low_missing, "best_sim": None}, 0.55) is None

def test_hard_gates_never_apply_to_follow_up_turns(monkeypatch):
    from test_answer import Scripted
    import json
    monkeypatch.setitem(answer.DEFAULTS, "sufficiency", "hard_sim_anchor")
    reply = json.dumps({"s": [{"t": "ARM grew from RISC.", "c": ["S1"], "k": "src"}]})
    llm = Scripted(reply, "stop")
    out = answer.generate("Which of the two is described as a duopoly?", pack(0.3), llm=llm, timeout=5,
                          follow_up=True)
    assert llm.calls == 1 and out["status"] == "ok" and out["gate"]["follow_up"] and not out["gate"]["abstained"]
    llm = Scripted(reply, "stop")
    out = answer.generate("Which of the two is described as a duopoly?", pack(0.3), llm=llm, timeout=5)
    assert llm.calls == 0 and out["status"] == "insufficient_evidence"