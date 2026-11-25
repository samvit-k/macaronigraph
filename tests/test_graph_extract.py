"""Graph extraction without an LLM: input preparation, JSON parsing, mention matching and grounding."""
import pytest

from backend.graph import extract as ex
from test_chunking import make_record, oseg, seg

def prep(*texts, context=()):
    return {"reel_id": "R1", "content_hash": "h",
            "passages": [{"pid": f"P{i + 1}", "chunk_id": f"R1:transcript:{i:03d}", "content_type": "transcript",
                          "text": t, "start": float(i), "end": float(i + 1)} for i, t in enumerate(texts)],
            "context": list(context)}

def concept(name, surface=None, passages=("P1",), type_="technology"):
    return {"name": name, "type": type_, "surface": surface or name, "passages": list(passages)}

def rel(s, p, o, pid, quote):
    return {"subject": s, "predicate": p, "object": o, "passage": pid, "quote": quote}

def test_prepare_excludes_fragments_and_makes_short_overlays_context_only():
    rec = make_record([seg(0, 5, "Triton is built on top of CUDA for writing GPU kernels.")],
                      [oseg(0, 28, "SHIP IT"), oseg(3, 4, "aN", quality="fragment"),
                       oseg(5, 6, "Step one install the CUDA toolkit")], duration=30)
    p = ex.prepare(rec)
    kinds = [x["content_type"] for x in p["passages"]]
    assert "ocr_fragment" not in kinds and p["context"] == ["SHIP IT"]
    assert all(x["chunk_id"].startswith("R1:") for x in p["passages"])
    assert "SHIP IT" not in "\n".join(x["text"] for x in p["passages"])
    prompt = ex.prompt_for(p)
    assert "CONTEXT ONLY" in prompt and "\n1. (" in prompt and "SHIP IT" in prompt.split("CONTEXT ONLY")[1]

def test_parse_json_requires_both_lists_but_allows_empty_ones():
    assert ex.parse_json('{"concepts": [], "relations": []}') == {"concepts": [], "relations": []}
    assert ex.parse_json('```json\n{"concepts": [{"name": "x"}], "relations": []}\n```')["concepts"]
    for bad in ('{"concepts": []}', '{"relations": []}', '{}', '{"concepts": {}, "relations": []}',
                '[1, 2]', 'no json here', '{"concepts": [}'):
        with pytest.raises(ex.ExtractionError):
            ex.parse_json(bad)

def test_parse_compact_positional_format():
    out = ex.parse_json('{"c":[["Triton","technology","Triton",[2]],["CUDA","technology","CUDA",["3"]],["bad"]],'
                        '"r":[["Triton","DEPENDS_ON","CUDA",2,"Triton is built on top of CUDA"],[1,2]]}')
    assert out["concepts"][0] == {"name": "Triton", "type": "technology", "surface": "Triton", "passages": ["P2"]}
    assert out["concepts"][1]["passages"] == ["P3"] and out["concepts"][2] is None
    assert out["relations"][0]["passage"] == "P2" and out["relations"][1] is None
    for bad in ('{"c": []}', '{"r": []}', '{"c": {}, "r": []}'):
        with pytest.raises(ex.ExtractionError):
            ex.parse_json(bad)

def test_compact_rows_ground_end_to_end():
    raw = ex.parse_json('{"c":[["Triton","technology","Triton",[1]],["CUDA","technology","CUDA",[1]]],'
                        '"r":[["Triton","DEPENDS_ON","CUDA",1,"Triton is built on top of CUDA"]]}')
    g = ex.ground(raw, prep("Triton is built on top of CUDA for writing fast GPU kernels."))
    assert [(a["subject"], a["predicate"], a["object"]) for a in g["assertions"]] == [("triton", "DEPENDS_ON", "cuda")]

class _FakeLLM:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def generate(self, system, user, max_tokens=0):
        self.calls.append(user)
        text, reason = self.outputs.pop(0)
        return text, {"finish_reason": reason}

def test_truncated_output_is_retried_then_reported_not_silently_accepted():
    p = prep("Triton is built on top of CUDA.")
    ok = '{"c":[["Triton","technology","Triton",[1]]],"r":[]}'
    raw, stats = ex.propose(_FakeLLM(('{"c":[["Tri', "length"), (ok, "stop")), p)
    assert raw["concepts"][0]["name"] == "Triton" and "truncated" in stats["retried"]
    with pytest.raises(ex.ExtractionError, match="truncated"):
        ex.propose(_FakeLLM(('{"c":[', "length"), ('{"c":[', "length")), p)
    with pytest.raises(ex.ExtractionError):
        ex.propose(_FakeLLM(("not json", "stop"), ("still not json", "stop")), p)

@pytest.mark.parametrize("surface,text", [("CUDA", "Install CUDA-Q today"), ("x86", "built for x86-64 chips"),
                                          ("CUDA", "the CUDAQ platform"), ("C", "written in C++"), ("C", "or C#"),
                                          ("C++", "plain C code"), ("GPU", "a TPU is different"),
                                          ("x86-64", "x86 only")])
def test_distinct_identifiers_never_match(surface, text):
    assert ex.surface_match(surface, text) is None

@pytest.mark.parametrize("surface,text,kind", [
    ("retrieval-augmented generation", "we use retrieval augmented generation here", "exact"),
    ("retrieval augmented generation", "Retrieval-Augmented Generation (RAG)", "exact"),
    ("CUDA-Q", "Step 1. Install CUDA-Q, NVIDIA's platform", "exact"),
    ("C++", "learn C++, then Rust", "exact"),
    ("agents.md", "put rules in your agents.md file.", "exact"),
    ("inference engineering", "why inferenoe engineering matters", "fuzzy"),
])
def test_legitimate_matches(surface, text, kind):
    assert ex.surface_match(surface, text) == kind

def test_fuzzy_matching_is_conservative():
    assert ex.surface_match("harness", "knife VS hamess VS oven") is None
    assert ex.surface_match("Python", "Pythons") is None
    assert ex.surface_match("printed circuit board", "printed circuit boards") is None
    assert ex.surface_match("CUDA", "CUDAQ") is None

def test_curated_plural_synonym_is_recognized_through_grounding():
    p = prep("Design your first PCB in KiCad.", "Then order printed circuit boards from a fab.")
    g = ex.ground({"concepts": [concept("Printed Circuit Board", "PCB", ["P1"])], "relations": []}, p)
    chunks = {m["chunk_id"]: m for m in g["mentions"] if m["key"] == "printed circuit board"}
    assert set(chunks) == {"R1:transcript:000", "R1:transcript:001"}
    assert chunks["R1:transcript:001"]["method"] == "string_match"
    assert chunks["R1:transcript:001"]["surface"] == "printed circuit boards"

def test_supported_relation_survives_as_proposed_with_provenance():
    p = prep("Triton is built on top of CUDA for writing fast GPU kernels.")
    raw = {"concepts": [concept("Triton"), concept("CUDA"), concept("GPU", "GPU")],
           "relations": [rel("Triton", "DEPENDS_ON", "CUDA", "P1", "Triton is built on top of CUDA")]}
    g = ex.ground(raw, p)
    (a,) = g["assertions"]
    assert (a["subject"], a["predicate"], a["object"]) == ("triton", "DEPENDS_ON", "cuda")
    assert a["status"] == "proposed" and a["provenance"] == "quote_verified" and a["confidence"] == "quote_names_both"
    assert a["chunk_id"] == "R1:transcript:000" and a["reel_id"] == "R1"
    assert a["id"] == ex.assertion_id("R1", "R1:transcript:000", "triton", "DEPENDS_ON", "cuda")
    assert {m["key"] for m in g["mentions"]} == {"triton", "cuda", "gpu"}

def test_llm_plural_in_p1_and_singular_alias_in_p2_supports_relation_in_p2():
    p = prep("LLMs are everywhere right now.", "An LLM uses a Vector Database to retrieve context")
    raw = {"concepts": [concept("Large Language Model", "LLMs", ["P1"]),
                        concept("Vector Database", "Vector Database", ["P2"])],
           "relations": [rel("Large Language Model", "USES", "Vector Database", "P2",
                             "An LLM uses a Vector Database to retrieve context")]}
    g = ex.ground(raw, p)
    llm_key = "large language model"
    assert {(m["chunk_id"], m["key"]) for m in g["mentions"]} >= {
        ("R1:transcript:000", llm_key), ("R1:transcript:001", llm_key), ("R1:transcript:001", "vector database")}
    rec = next(m for m in g["mentions"] if m["chunk_id"] == "R1:transcript:001" and m["key"] == llm_key)
    assert rec["method"] == "string_match" and rec["surface"] == "LLM"
    (a,) = g["assertions"]
    assert (a["subject"], a["predicate"], a["object"], a["chunk_id"]) == (llm_key, "USES", "vector database",
                                                                          "R1:transcript:001")

def test_distinct_technologies_stay_distinct_through_grounding():
    p = prep("CUDA-Q runs quantum kernels, while CUDA runs classic GPU kernels.", "Learn C++ before C# and C.",
             "Compile for x86-64 or plain x86 targets.")
    names = ["CUDA-Q", "CUDA", "C++", "C#", "C", "x86-64", "x86"]
    g = ex.ground({"concepts": [concept(n, passages=["P1", "P2", "P3"]) for n in names], "relations": []}, p)
    assert len(g["concepts"]) == len(names)
    by_key = {}
    for m in g["mentions"]:
        by_key.setdefault(m["key"], set()).add(m["chunk_id"])
    assert by_key["cuda"] == by_key["cuda-q"] == {"R1:transcript:000"}
    assert by_key["c"] == by_key["c++"] == by_key["c#"] == {"R1:transcript:001"}
    assert by_key["x86"] == by_key["x86-64"] == {"R1:transcript:002"}

def test_concept_whose_surface_only_appears_as_a_different_identifier_is_rejected():
    g = ex.ground({"concepts": [concept("CUDA")], "relations": []}, prep("Install CUDA-Q, NVIDIA's platform"))
    assert g["concepts"] == {} and g["rejected"][0]["reason"].startswith("surface not found")

def test_unsupported_model_rename_is_not_used_to_resolve_relations():
    p = prep("Trident is built on top of CUDA for kernels.", "Triton is something else entirely and also uses CUDA")
    raw = {"concepts": [concept("Triton", "Trident", ["P1"]), concept("CUDA", passages=["P1"])],
           "relations": [rel("Triton", "DEPENDS_ON", "CUDA", "P1", "Trident is built on top of CUDA")]}
    g = ex.ground(raw, p)
    assert {"trident", "triton"} <= set(g["concepts"])
    by = {(m["chunk_id"], m["key"]) for m in g["mentions"]}
    assert ("R1:transcript:000", "trident") in by and ("R1:transcript:000", "triton") not in by
    assert any(r["kind"] == "rename" for r in g["rejected"])
    assert g["assertions"] == []
    assert any(r["kind"] == "relation" and "not both mentioned" in r["reason"] for r in g["rejected"])

def test_unsupported_rename_whose_name_is_absent_cannot_resolve_relations():
    p = prep("Trident is built on top of CUDA for kernels.")
    raw = {"concepts": [concept("Triton", "Trident", ["P1"]), concept("CUDA", passages=["P1"])],
           "relations": [rel("Triton", "DEPENDS_ON", "CUDA", "P1", "Trident is built on top of CUDA")]}
    g = ex.ground(raw, p)
    assert "trident" in g["concepts"] and "triton" not in g["concepts"]
    assert g["assertions"] == [] and any("not a grounded concept" in r["reason"] for r in g["rejected"])

def test_phrase_surfaces_never_become_concepts():
    p = prep("TheDemoApp embeds the Claude SDK under the hood.")
    raw = {"concepts": [concept("Claude SDK", "TheDemoApp embeds the Claude SDK", ["P1"], "tool"),
                        concept("Grok", "Grok relies on a small custom wrapper", ["P1"], "tool")], "relations": []}
    g = ex.ground(raw, p)
    assert set(g["concepts"]) == {"claude sdk"}
    assert all(len(c["name"].split()) <= ex.MAX_SURFACE_WORDS for c in g["concepts"].values())

def test_guarded_asr_correction_is_accepted():
    p = prep("Pick a terminal assistant such as Cloud Code for refactors.")
    g = ex.ground({"concepts": [concept("Claude Code", "Cloud Code", type_="tool")], "relations": []}, p)
    assert "claude code" in g["concepts"] and g["concepts"]["claude code"]["name"] == "Claude Code"

@pytest.mark.parametrize("relation,reason", [
    (rel("Triton", "INVENTED_BY", "CUDA", "P1", "Triton is built on top of CUDA"), "predicate not in vocabulary"),
    (rel("Triton", "DEPENDS_ON", "CUDA", "P1", "Triton is basically powered by CUDA"), "quote not found verbatim"),
    (rel("Triton", "DEPENDS_ON", "CUDA", "P9", "Triton is built on top of CUDA"), "not an evidence passage"),
    (rel("Triton", "DEPENDS_ON", "CUDA", "P1", "Triton"), "quote length"),
    (rel("Triton", "DEPENDS_ON", "PyTorch", "P1", "Triton is built on top of CUDA"), "not a grounded concept"),
    (rel("Triton", "DEPENDS_ON", "Triton", "P1", "Triton is built on top of CUDA"), "self-relation"),
])
def test_bad_relations_are_rejected_with_reasons(relation, reason):
    p = prep("Triton is built on top of CUDA for writing fast GPU kernels.")
    g = ex.ground({"concepts": [concept("Triton"), concept("CUDA")], "relations": [relation]}, p)
    assert g["assertions"] == []
    assert any(r["kind"] == "relation" and reason in r["reason"] for r in g["rejected"]), g["rejected"]

def test_relation_requires_both_concepts_in_the_cited_passage():
    p = prep("Triton writes GPU kernels quickly.", "CUDA is the base layer for Triton kernels and more")
    raw = {"concepts": [concept("Triton", passages=["P1"]), concept("CUDA", passages=["P2"])],
           "relations": [rel("Triton", "DEPENDS_ON", "CUDA", "P1", "Triton writes GPU kernels quickly")]}
    g = ex.ground(raw, p)
    assert g["assertions"] == [] and any("not both mentioned" in r["reason"] for r in g["rejected"])

def test_hallucinated_and_generic_concepts_are_rejected_and_overlays_are_not_evidence():
    p = prep("This video has tips for students.", context=["SHIP IT", "TPU"])
    g = ex.ground({"concepts": [concept("TPU", passages=["P1"]), concept("tips"), concept("Kubernetes")],
                   "relations": []}, p)
    assert g["concepts"] == {}
    reasons = {r["item"]: r["reason"] for r in g["rejected"]}
    assert reasons["tips"] == "generic" and reasons["TPU"].startswith("surface not found")
    assert reasons["Kubernetes"].startswith("surface not found")

def test_grounding_is_deterministic():
    p = prep("Triton is built on top of CUDA for writing fast GPU kernels.")
    raw = {"concepts": [concept("Triton"), concept("CUDA")],
           "relations": [rel("Triton", "DEPENDS_ON", "CUDA", "P1", "Triton is built on top of CUDA")]}
    assert ex.ground(raw, p) == ex.ground(raw, p)

def test_wrong_citation_is_only_a_hint():
    p = prep("Thank you.", "STM32 board= ctrl: Nav2 + route + mapping")
    g = ex.ground({"concepts": [concept("Nav2", passages=["P1"]), concept("STM32", passages=["P1"])],
                   "relations": []}, p)
    assert set(g["concepts"]) == {"nav2", "stm32"}
    assert {m["chunk_id"] for m in g["mentions"]} == {"R1:transcript:001"}

def test_relation_quote_must_name_both_concepts():
    p = prep("A chef needs tools like knives and Websearch. A harness keeps the tools near the stove.")
    raw = {"concepts": [concept("harness"), concept("Websearch")],
           "relations": [rel("harness", "USES", "Websearch", "P1", "A chef needs tools like knives and Websearch")]}
    g = ex.ground(raw, p)
    assert g["assertions"] == [] and any("does not name both concepts exactly" in r["reason"] for r in g["rejected"])

def test_uncorroborated_ocr_surface_is_rejected_but_spoken_one_kept():
    ocr_only = {"reel_id": "R1", "content_hash": "h", "context": [],
                "passages": [{"pid": "P1", "chunk_id": "R1:ocr:000", "content_type": "ocr", "text": "vect Sea O1 ranking",
                              "start": 0.0, "end": 1.0}]}
    g = ex.ground({"concepts": [concept("vector search", "vect Sea")], "relations": []}, ocr_only)
    assert g["concepts"] == {} and any("uncorroborated OCR" in r["reason"] for r in g["rejected"])
    spoken = prep("We use Trident for kernels.")
    g = ex.ground({"concepts": [concept("Triton", "Trident")], "relations": []}, spoken)
    assert "trident" in g["concepts"]

def test_underscored_names_get_readable_display_names():
    g = ex.ground({"concepts": [concept("Pi_harness", "Pi harness")], "relations": []},
                  prep("The Pi harness ships with the demo kit."))
    assert g["concepts"]["pi harness"]["name"] == "Pi harness"

class _BudgetLLM:

    def __init__(self, *outputs):
        self.outputs, self.budgets = list(outputs), []

    def generate(self, system, user, max_tokens=0):
        self.budgets.append(max_tokens)
        text, reason = self.outputs.pop(0)
        return text, {"finish_reason": reason, "seconds": 1.5, "generation_tokens": 10}

MISSING_BRACES = ('{"c":[{"n":"Triton","t":"technology","s":"Triton","p":[1]},"n":"CUDA","t":"technology","s":"CUDA",'
                  '"p":[1]}],"r":[{"s":"Triton","p":"DEPENDS_ON","o":"CUDA","i":1,"q":"Triton is built on top of '
                  'CUDA"},"s":"CUDA","p":"USED_FOR","o":"Triton","i":1,"q":"Triton is built on top of CUDA"}]}')

def test_structural_repair_fixes_missing_braces_without_adding_content():
    with pytest.raises(ex.ExtractionError):
        ex.parse_json(MISSING_BRACES)
    data, repairs = ex.parse_with_repair(MISSING_BRACES)
    assert [c["name"] for c in data["concepts"]] == ["Triton", "CUDA"] and len(data["relations"]) == 2
    assert repairs and "missing '{'" in repairs[0]
    fixed, _ = ex.repair_json(MISSING_BRACES)
    strip = lambda s: "".join(ch for ch in s if ch not in "{}[],")  # noqa: E731
    assert strip(fixed) == strip(MISSING_BRACES)

def test_repair_keeps_strict_schema_validation():
    for bad in ('{"c":[{"n":"x"},"n":"y"}]}', '{"r":[]}', 'nothing', '{"c":[{"n":"x"}],"r":[{"s":}]}'):
        with pytest.raises(ex.ExtractionError):
            ex.parse_with_repair(bad)

def test_repaired_first_attempt_needs_no_retry():
    llm = _BudgetLLM((MISSING_BRACES, "stop"))
    raw, stats = ex.propose(llm, prep("Triton is built on top of CUDA."))
    assert llm.budgets == [ex.MAX_NEW_TOKENS] and stats["repairs"] and "retried" not in stats
    assert stats["attempts"][0]["outcome"] == "repaired"

def test_truncated_first_attempt_is_retried_with_a_larger_budget_and_timed():
    ok = '{"c":[{"n":"Triton","t":"technology","s":"Triton","p":[1]}],"r":[]}'
    llm = _BudgetLLM(('{"c":[{"n":"Tri', "length"), (ok, "stop"))
    raw, stats = ex.propose(llm, prep("Triton is built on top of CUDA."))
    assert llm.budgets == [ex.MAX_NEW_TOKENS, ex.RETRY_MAX_TOKENS] and ex.RETRY_MAX_TOKENS > ex.MAX_NEW_TOKENS
    assert [a["outcome"] for a in stats["attempts"]] == ["truncated", "ok"] and stats["seconds"] == 3.0
    assert raw["concepts"][0]["name"] == "Triton" and not stats.get("partial")

def test_truncated_retry_keeps_only_fully_emitted_objects_and_is_flagged_partial():
    cut = ('{"c":[{"n":"Triton","t":"technology","s":"Triton","p":[1]},{"n":"CUDA","t":"technology","s":"CUDA",'
           '"p":[1]},{"n":"Pyt')
    llm = _BudgetLLM((cut, "length"), (cut, "length"))
    raw, stats = ex.propose(llm, prep("Triton is built on top of CUDA."))
    assert [c["name"] for c in raw["concepts"]] == ["Triton", "CUDA"]
    assert raw["relations"] == [] and stats["partial"] is True
    assert stats["salvage"]["r_list_missing"] and not stats["salvage"]["c_list_complete"]

def test_salvage_never_completes_a_cut_relation():
    cut = ('{"c":[{"n":"Triton","t":"technology","s":"Triton","p":[1]}],"r":[{"s":"Triton","p":"USES","o":"CUDA",'
           '"i":1,"q":"Triton is built on top of CUDA"},{"s":"Triton","p":"USES","o":"Py')
    data, info = ex.salvage(cut)
    assert len(data["relations"]) == 1 and info["c_list_complete"] and not info["r_list_complete"]

def test_nothing_salvageable_is_an_error_carrying_attempt_timings():
    llm = _BudgetLLM(('{"c":[', "length"), ('{"c":[{"n":"Tr', "length"))
    with pytest.raises(ex.ExtractionError) as e:
        ex.propose(llm, prep("Triton is built on top of CUDA."))
    assert [a["outcome"] for a in e.value.attempts] == ["truncated", "truncated"]

def test_r1_label_is_reproduced_and_r2_is_a_distinct_version():
    r1 = ex._r1_version("m", "rev")
    assert ex.compatible_versions("m", "rev") == [r1] and ex.extractor_version("m", "rev") != r1
    assert ex.extractor_version("m", "rev").startswith("x-p4i1r2-") and r1.startswith("x-p4i1-")

def _ground_rel(text, a, b, pred, quote=None):
    raw = {"concepts": [concept(a), concept(b)], "relations": [rel(a, pred, b, "P1", quote or text)]}
    return ex.ground(raw, prep(text))

@pytest.mark.parametrize("text,a,b,reason", [
    ("Topics: drivers, computer vision, and SLAM for a hobby rover.", "computer vision", "SLAM",
     "enumeration"),
    ("Watch for injection, authentication, or SSRF in your APIs.", "injection", "SSRF", "enumeration"),
    ("I recommend Python. You should also try LeetCode problems daily.", "Python", "LeetCode", "different sentences"),
])
def test_lists_and_cross_sentence_quotes_are_not_relations(text, a, b, reason):
    g = _ground_rel(text, a, b, "USES")
    assert g["assertions"] == [] and any(reason in r["reason"] for r in g["rejected"]), g["rejected"]

@pytest.mark.parametrize("text,a,b,pred", [
    ("ARM vs x86, which architecture is winning?", "ARM", "x86", "CONTRASTS_WITH"),
    ("PyTorch and JAX are alternatives for training models.", "PyTorch", "JAX", "ALTERNATIVE_TO"),
    ("Triton is built on top of CUDA for kernels.", "Triton", "CUDA", "DEPENDS_ON"),
    ("Whisperflow costs money, and Handy is a free alternative to it.", "Whisperflow", "Handy",
     "ALTERNATIVE_TO"),
])
def test_stated_relations_and_comparisons_survive(text, a, b, pred):
    assert len(_ground_rel(text, a, b, pred)["assertions"]) == 1

def test_fuzzy_spelling_never_counts_as_relation_evidence():
    text = "Grind LeetCode daily. Python with the NeetCode roadmap is the holy grail."
    g = ex.ground({"concepts": [concept("Python"), concept("LeetCode")],
                   "relations": [rel("Python", "USES", "LeetCode", "P1", "Python with the NeetCode roadmap")]}, prep(text))
    assert g["assertions"] == [] and any("exactly" in r["reason"] for r in g["rejected"])

def test_neetcode_and_leetcode_are_distinct_concepts():
    g = ex.ground({"concepts": [concept("LeetCode", "NeetCode")], "relations": []}, prep("Watch NeetCode videos."))
    assert "leetcode" not in g["concepts"]

def test_several_predicates_for_one_statement_keep_none():
    text = "Inference engineers use quantization and caching to cut costs."
    raw = {"concepts": [concept("Inference engineers"), concept("quantization")],
           "relations": [rel("Inference engineers", p, "quantization", "P1", text)
                         for p in ("DEPENDS_ON", "PREREQUISITE_FOR")]}
    g = ex.ground(raw, prep(text))
    assert g["assertions"] == [] and any("several predicates" in r["reason"] for r in g["rejected"])

def test_name_inside_another_name_is_not_a_relation():
    text = 'Add "args": ["@playwright/mcp@latest"] to the config.'
    raw = {"concepts": [concept("Playwright"), concept("Playwright MCP", "@playwright/mcp@latest")],
           "relations": [rel("Playwright", "USES", "Playwright MCP", "P1", '"args": ["@playwright/mcp@latest"]')]}
    g = ex.ground(raw, prep(text))
    assert g["assertions"] == [] and any("inside the other's name" in r["reason"] for r in g["rejected"])

def test_numeric_values_are_not_concepts():
    g = ex.ground({"concepts": [concept("1, 6 and 11"), concept("$180K"), concept("2.4 GHz")], "relations": []},
                  prep("Use channels 1, 6 and 11 on 2.4 GHz; salaries reach $180K."))
    assert set(g["concepts"]) == {"2.4 ghz"}

def test_short_ocr_tokens_are_rejected_unless_curated_or_spoken():
    p = prep("GE", "C is fast")
    for x in p["passages"]:
        x["content_type"] = "ocr"
    g = ex.ground({"concepts": [concept("GE"), concept("C", passages=("P2",))], "relations": []}, p)
    assert set(g["concepts"]) == {"c"}
    spoken = ex.ground({"concepts": [concept("Go")], "relations": []}, prep("I write services in Go"))
    assert "go" in spoken["concepts"]

def test_curated_fixes_klayout_neetcode_esp32_playwright_mcp_common_coder():
    g = ex.ground({"concepts": [concept("K-Layout"), concept("KLayout", passages=("P2",)),
                                concept("NEETCODE", passages=("P2",)), concept("ESP32", type_="concept"),
                                concept("@pLaywright/mcp@latest", passages=("P2",)), concept("Common Coder")],
                   "relations": []},
                  prep("K-Layout designs chips on an ESP32 board. Common Coder joins.",
                       'KLayout and NEETCODE, "args": ["@pLaywright/mcp@latest"]'))
    by = {c["name"]: c for c in g["concepts"].values()}
    assert {"KLayout", "NeetCode", "ESP32", "Playwright MCP"} <= set(by) and "Common Coder" not in by
    assert set(by["KLayout"]["surfaces"]) == {"K-Layout", "KLayout"} and len(g["concepts"]) == 4
    assert by["NeetCode"]["type"] == "resource" and by["ESP32"]["type"] == "technology"
    assert any("curated non-concept" in r["reason"] for r in g["rejected"])

def test_higgs_field_with_company_context_maps_to_higgsfield_keeping_surface_and_provenance():
    p = prep("Open the plugin list and enable the Higgs Field MCP plugin before uploading an image.")
    g = ex.ground({"concepts": [concept("Higgs Field", type_="organization")], "relations": []}, p)
    assert set(g["concepts"]) == {"higgsfield"} and g["concepts"]["higgsfield"]["name"] == "Higgsfield"
    assert g["concepts"]["higgsfield"]["type"] == "organization"
    [m] = g["mentions"]
    assert m["surface"] == "Higgs Field" and m["method"] == "llm+exact_match+context_alias"
    assert m["chunk_id"] == p["passages"][0]["chunk_id"]

@pytest.mark.parametrize("text", [
    "The Higgs field gives elementary particles their mass, confirmed at CERN in 2012.",
    "AI explains physics: the Higgs field gives particles mass.",
])
def test_higgs_field_in_physics_stays_a_distinct_scientific_concept(text):
    g = ex.ground({"concepts": [concept("Higgs field", type_="concept")], "relations": []}, prep(text))
    assert set(g["concepts"]) == {"higgs field"} and "higgsfield" not in g["concepts"]
    assert g["mentions"][0]["method"] == "llm+exact_match"

def test_one_reel_with_both_meanings_keeps_each_passage_with_its_own_concept():
    p = prep("Enable the Higgs Field MCP plugin inside ChatGPT.", "In physics the Higgs field gives particles mass.")
    g = ex.ground({"concepts": [concept("Higgs Field", passages=("P1", "P2"))], "relations": []}, p)
    by_chunk = {m["chunk_id"]: m["key"] for m in g["mentions"]}
    assert by_chunk == {p["passages"][0]["chunk_id"]: "higgsfield", p["passages"][1]["chunk_id"]: "higgs field"}

def test_context_alias_recall_does_not_cross_contexts():
    p = prep("Enable the Higgs Field MCP plugin.", "Separately, the Higgs field gives mass.")
    g = ex.ground({"concepts": [concept("Higgs Field", passages=("P1",))], "relations": []}, p)
    assert [(m["key"], m["chunk_id"]) for m in g["mentions"]] == [("higgsfield", p["passages"][0]["chunk_id"])]

def test_context_rule_is_not_an_unconditional_alias():
    from backend.graph import normalize as nz
    assert nz.canonicalize("Higgs Field")[0] == "higgs field"
    assert frozenset(("higgs field", "higgsfield")) in nz.contextual_pairs()
    assert not nz.context_supports(nz.context_alias("Higgs Field"), "AI video tools and AI models")