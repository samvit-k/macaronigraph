"""Phase 4C citation parsing (backend/answer.py, inline_citations): markers inside sentences, per-sentence
association, malformed/unknown ids, and the unchanged evidence requirements. Scripted model; no real LLM."""
import pytest

from backend import answer
from test_answer import ask


@pytest.fixture(autouse=True)
def inline(isolated_output, monkeypatch):
    monkeypatch.setitem(answer.DEFAULTS, "inline_citations", True)


def sent(t, c=None, k="src"):
    return {"t": t, "c": c if c is not None else [], "k": k}


def test_inline_marker_inside_a_sentence_is_a_valid_citation():
    out, _ = ask({"s": [sent("ARM grew from RISC [S1] and has better battery life.")]})
    assert out["status"] == "ok" and out["sentences"][0]["citations"] == ["S1"]
    assert "[S1] and" not in out["sentences"][0]["text"] and out["sentences"][0]["text"].endswith("battery life.")


def test_c_prefixed_marker_previously_removed_is_now_parsed():
    # Phase 4B failure pattern: citations written as text ("c:[S2]") with an empty "c" → every sentence removed.
    out, _ = ask({"s": [sent("The Reel says x86 grew from CISC, c:[S2]")]})
    assert out["status"] == "ok" and out["sentences"][0]["citations"] == ["S2"]
    assert "c:" not in out["sentences"][0]["text"]


def test_without_the_flag_the_same_output_is_removed():
    answer.DEFAULTS["inline_citations"] = False
    out, _ = ask({"s": [sent("The Reel says x86 grew from CISC, c:[S2]")]})
    assert out["status"] == "insufficient_evidence"


def test_multiple_markers_in_one_sentence_are_all_kept():
    out, _ = ask({"s": [sent("ARM grew from RISC [S1] while x86 grew from CISC [S2].")]})
    assert out["sentences"][0]["citations"] == ["S1", "S2"]
    out, _ = ask({"s": [sent("ARM grew from RISC while x86 grew from CISC [S1, S2].")]})
    assert out["sentences"][0]["citations"] == ["S1", "S2"]


def test_each_sentence_keeps_only_its_own_markers():
    out, _ = ask({"s": [sent("ARM grew from RISC [S1]. x86 grew from CISC [S2].", ["S1", "S2"])]})
    assert [(s["text"], s["citations"]) for s in out["sentences"]] == [("ARM grew from RISC.", ["S1"]),
                                                                       ("x86 grew from CISC.", ["S2"])]


def test_marker_after_the_full_stop_belongs_to_the_preceding_sentence():
    out, _ = ask({"s": [sent("ARM grew from RISC. [S1] x86 grew from CISC. [S2]")]})
    assert [s["citations"] for s in out["sentences"]] == [["S1"], ["S2"]]


def test_a_sentence_without_its_own_marker_never_borrows_a_neighbours_citation():
    out, _ = ask({"s": [sent("ARM grew from RISC [S1]. Quantum chips beat both.", ["S1", "S2"])]})
    assert [s["text"] for s in out["sentences"]] == ["ARM grew from RISC."]
    assert any(r["text"] == "Quantum chips beat both." and "no valid citation" in r["reason"] for r in out["removed"])


def test_unknown_and_malformed_identifiers_are_rejected_and_reported():
    out, _ = ask({"s": [sent("ARM grew from RISC [S9]."), sent("x86 grew from CISC [2]."),
                        sent("ARM has better battery life [Source 1]."), sent("ARM is simple [S0].")]})
    assert out["status"] == "insufficient_evidence" and out["sentences"] == []
    reasons = " | ".join(r["reason"] for r in out["removed"])
    assert "S9, not in the sources" in reasons
    assert "malformed citation '[2]'" in reasons and "malformed citation '[Source 1]'" in reasons
    assert "malformed citation '[S0]'" in reasons


def test_unsupported_sentence_with_an_inline_citation_is_still_only_flagged_not_trusted():
    out, _ = ask({"s": [sent("Quantum annealing outperforms both on cryptography [S1].")]})
    assert out["sentences"][0]["support"] == "weak"                     # diagnostic flag, as before


def test_inline_citations_do_not_bypass_quote_validation():
    out, _ = ask({"s": [sent("The Reel says 'ARM always beats x86 everywhere' [S1].")]})
    assert out["status"] == "insufficient_evidence" and "quote not found" in out["removed"][0]["reason"]
    out, _ = ask({"s": [sent("The Reel says 'x86 grew from CISC' [S1].")]})          # real quote, wrong source
    assert out["status"] == "insufficient_evidence"
    out, _ = ask({"s": [sent("The Reel says 'x86 grew from CISC' [S2].")]})
    assert out["status"] == "ok"


def test_inference_with_an_inline_citation_keeps_its_kind():
    out, _ = ask({"s": [sent("So ARM laptops likely last longer [S1].", k="inf")]})
    assert out["sentences"][0]["kind"] == "inference" and out["sentences"][0]["citations"] == ["S1"]


def test_rendered_answer_has_no_duplicate_or_dangling_markers():
    out, _ = ask({"s": [sent("ARM grew from RISC [S1].", ["S1"])]})
    assert out["text"].count("[S1]") == 1 and "[S1]." not in out["text"].split("[S1]", 1)[0]


def test_abbreviations_and_decimals_do_not_split_a_sentence():
    parts, _ = answer.split_inline("Tools, e.g. Arduino, cost 0.74 dollars vs. kits [S1]. x86 is CISC [S2].", [])
    assert [c for _, c in parts] == [["S1"], ["S2"]] and parts[0][0].startswith("Tools, e.g. Arduino")


def test_a_single_trailing_marker_covers_the_whole_text_as_one_unit():
    # Dev failure (v1 m04.t2): a multi-sentence quotation followed by one "(S2)" was split and its first sentences
    # removed. Markers only at the very end behave like the item's "c": one unit, checked as a whole.
    out, _ = ask({"s": [sent("The Reel says ARM grew from RISC. It also says x86 grew from CISC. (S1)", ["S1"])]})
    assert out["status"] == "ok" and len(out["sentences"]) == 1 and not out["removed"]
    assert out["sentences"][0]["citations"] == ["S1"] and "(S1)" not in out["sentences"][0]["text"]


def test_interior_markers_still_split_and_never_borrow():
    out, _ = ask({"s": [sent("ARM grew from RISC [S1]. Quantum chips beat both. x86 grew from CISC [S2].")]})
    assert [s["citations"] for s in out["sentences"]] == [["S1"], ["S2"]]
    assert any(r["text"] == "Quantum chips beat both." for r in out["removed"])


def test_json_style_quoted_ids_in_a_text_marker_are_parsed():
    # Dev failure (v2 d14): '<a sentence with c:["S1","S2"] inside its text>' was removed.
    out, _ = ask({"s": [sent('ARM grew from RISC, c:["S1","S2"]')]})
    assert out["status"] == "ok" and out["sentences"][0]["citations"] == ["S1", "S2"]
    assert "c:" not in out["sentences"][0]["text"] and '"' not in out["sentences"][0]["text"]
    out, _ = ask({"s": [sent("ARM grew from RISC ['S1'].")]})
    assert out["sentences"][0]["citations"] == ["S1"]
