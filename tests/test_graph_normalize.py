"""Concept normalization: genuine synonyms merge; related-but-distinct concepts stay separate."""
import json
import os

import pytest

from backend.graph import normalize as n


def key(name, **kw):
    return n.canonicalize(name, **kw)[0]


@pytest.mark.parametrize("group", [["C", "C++", "C#"], ["CUDA", "CUDA-Q"], ["x86", "x86-64"], ["GPU", "TPU", "NPU", "CPU"],
                                   ["Model Context Protocol", "MCP server"], ["Printed Circuit Board", "PCB design"],
                                   ["AI Agent", "LLM agent", "coding agent"], ["Vector Database", "vector store"],
                                   ["Claude", "Claude Code"]])
def test_related_but_distinct_concepts_stay_separate(group):
    keys = [key(g) for g in group]
    assert len(set(keys)) == len(group), dict(zip(group, keys))


def test_technical_identifiers_keep_meaningful_punctuation():
    assert (n.base_key("C++"), n.base_key("C#"), n.base_key("C")) == ("c++", "c#", "c")
    assert n.base_key("CUDA-Q") == "cuda-q" and n.base_key("x86-64") == "x86-64"
    assert n.base_key("Node.js") == "node.js" and n.base_key(".NET") == ".net"
    assert n.base_key("agents.md") == "agents.md"
    for t in ("C", "C++", "C#", "x86", "CUDA-Q"):
        assert not n.is_generic(t), t


@pytest.mark.parametrize("a,b", [("CUDAQ", "CUDA-Q"), ("CUDA Q", "CUDA-Q"), ("cpp", "C++"), ("C sharp", "C#"),
                                 ("x64", "x86-64"), ("AMD64", "x86-64"), ("ROS2", "ROS 2")])
def test_curated_identifier_synonyms_merge(a, b):
    assert key(a) == key(b)


@pytest.mark.parametrize("name", ["Pandas", "Windows", "Kubernetes", "Jenkins", "Robotics", "Electronics",
                                  "Linus Torvalds", "Johns Hopkins", "Google Docs"])
def test_proper_names_are_not_singularized(name):
    assert n.base_key(name) == name.lower()


def test_lowercase_multiword_plurals_still_singularize():
    assert n.base_key("neural networks") == n.base_key("Neural Network") == "neural network"


@pytest.mark.parametrize("a,b", [("vector databases", "Vector Database"), ("knowledge graphs", "Knowledge Graph"),
                                 ("GPUs", "GPU"), ("LLMs", "Large Language Model"), ("AI agents", "AI Agent"),
                                 ("coding agents", "Coding Agent"), ("RAG", "retrieval augmented generation"),
                                 ("retrieval-augmented generation", "Retrieval-Augmented Generation"),
                                 ("transformers", "transformer"), ("neural networks", "Neural Network"),
                                 ("SLAM", "Simultaneous Localization and Mapping")])
def test_ordinary_plurals_and_synonyms_merge(a, b):
    assert key(a) == key(b), (key(a), key(b))


def test_asr_aliases_only_apply_when_the_model_agrees():
    assert key("Cloud Code") == "cloud code"                                  # heard, no corroboration
    assert key("Cloud Code", model_canonical="Claude Code") == key("Claude Code")
    assert key("lead code", model_canonical="LeetCode") == key("LeetCode")
    assert key("Anthropik", model_canonical="Anthropic") == key("Anthropic")
    assert key("Anthropik", model_canonical="Anthropik") == "anthropik"


# ---------------------------------------------------------------- acronym definitions

def test_acronym_without_connector_words():
    acr = n.find_acronyms("We use proximal policy optimization (PPO) and model predictive control (MPC).")
    assert acr == {"ppo": "proximal policy optimization", "mpc": "model predictive control"}
    assert key("PPO", acronyms=acr) == key("proximal policy optimization")


def test_acronym_whose_letters_include_connector_words():
    acr = n.find_acronyms("Robots rely on Simultaneous Localization and Mapping (SLAM) to navigate.")
    assert acr == {"slam": "Simultaneous Localization and Mapping"}
    acr = n.find_acronyms("Track the bill of materials (BOM) and return on investment (ROI).")
    assert acr == {"bom": "bill of materials", "roi": "return on investment"}


def test_acronym_skipping_connector_words():
    acr = n.find_acronyms("the department of defense (DoD) funds it")
    assert acr == {"dod": "department of defense"}                            # initials incl. connector
    acr = n.find_acronyms("an internet of things (IoT) device")
    assert acr == {"iot": "internet of things"}
    acr = n.find_acronyms("United States of America (USA)")
    assert acr == {"usa": "United States of America"}                         # connector skipped


@pytest.mark.parametrize("text", ["I love pizza (NYC)", "see the docs (README) first", "Python is great (AI)",
                                  "the chief executive officer of the company (CEO)", "something (OK)"])
def test_unrelated_parentheticals_are_not_acronym_definitions(text):
    assert n.find_acronyms(text) == {}


# ---------------------------------------------------------------- generic filter

@pytest.mark.parametrize("w", ["video", "tips", "the best tools", "people", "free resources", "2026", "x",
                               "the", "all of the things", "this is a whole sentence that is not a concept at all"])
def test_generic_filler_is_rejected(w):
    assert n.is_generic(w)


@pytest.mark.parametrize("w", ["Triton", "inference engineering", "the Pi harness", "LTspice", "C", "R2D2"])
def test_real_concepts_are_not_generic(w):
    assert not n.is_generic(w)


# ---------------------------------------------------------------- alias file

def test_alias_file_updates_are_picked_up(tmp_path, monkeypatch):
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps({"concepts": {"Foo Bar": {"type": "tool", "aliases": ["FB"]}}}))
    monkeypatch.setattr(n, "ALIASES_PATH", p)
    v1 = n.version()
    assert key("FB") == "foo bar" and key("Fizz") == "fizz"
    p.write_text(json.dumps({"concepts": {"Foo Bar": {"type": "tool", "aliases": ["FB"]},
                                          "Fizz Buzz": {"type": "tool", "aliases": ["Fizz"]}}}))
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 1_000_000))   # ensure a new mtime
    assert key("Fizz") == "fizz buzz" and n.version() != v1


def test_real_alias_file_is_consistent():
    data = json.loads(n.ALIASES_PATH.read_text())["concepts"]
    seen = {}
    for canon, spec in data.items():
        assert spec["type"] in n.CONCEPT_TYPES
        for a in [canon, *spec.get("aliases", [])]:
            k = n.base_key(a)
            assert seen.setdefault(k, canon) == canon, f"{a!r} maps to both {seen[k]!r} and {canon!r}"
