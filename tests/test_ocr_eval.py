"""OCR evaluation report logic on synthetic results."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ocr_eval  # noqa: E402

WORDS = {"engineer", "explains", "agent", "setup", "the", "time"}

def seg(i, text):
    return {"id": i, "start": float(i), "end": float(i), "text": text}

def test_informative_tokens_keep_acronyms_numbers_and_recurring_names():
    raw = [seg(0, "RAG and AI agent"), seg(1, "x86 vs ARM"), seg(2, "GPT-4 setup"),
           seg(3, "Vultr GPU inference"), seg(4, "Vultr credits"), seg(5, "qzxv")]
    info = ocr_eval.informative_tokens(raw, WORDS, spoken=set())
    assert {"rag", "ai", "x86", "arm", "4", "gpu", "vultr", "agent", "setup"} <= info
    assert "qzxv" not in info and "credits" not in info

def test_retention_counts_technical_terms_not_just_dictionary_words():
    raw = [seg(0, "RAG pipeline x86"), seg(1, "RAG pipeline x86")]
    info = ocr_eval.informative_tokens(raw, WORDS, set())
    assert info == {"rag", "pipeline", "x86"}
    partial = ocr_eval.metrics([{"text": "RAG x86"}], info, WORDS, set())
    assert partial["retention"] == pytest.approx(2 / 3)
    full = ocr_eval.metrics([{"text": "RAG pipeline x86"}], info, WORDS, set())
    assert full["retention"] == 1.0

def test_missing_wordlist_is_graceful(tmp_path):
    assert ocr_eval.load_dict(tmp_path / "nope") is None
    m = ocr_eval.metrics([{"text": "AI agent"}], {"ai"}, None, set())
    assert m["readable"] is None and m["garbled"] is None and m["retention"] == 1.0

def test_missing_frames_reported_incomplete_and_nothing_evaluated_fails(capsys):
    def loader(rid):
        raise FileNotFoundError(rid)

    results, missing = ocr_eval.evaluate(["A", "B"], loader=loader)
    assert results == {} and missing == ["A", "B"]
    assert ocr_eval.report(results, missing, ["A", "B"], None, 0, None) == 1
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out and "NOTHING EVALUATED" in out and "PROXY METRICS" not in out

def test_partial_results_return_incomplete_code(capsys):
    fake = {"frames": 2, "raw": [], "filtered": [], "v1": [], "v2": [],
            "m1": ocr_eval.metrics([], set(), None, set()), "m2": ocr_eval.metrics([], set(), None, set()),
            "echo": {"speech_echo": 0, "partial_echo": 0, "on_screen": 0}}
    assert ocr_eval.report({"A": fake}, ["B"], ["A", "B"], None, 0, None) == 2
    assert "PROXY METRICS" in capsys.readouterr().out