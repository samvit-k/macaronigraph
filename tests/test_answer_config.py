"""The frozen answer configuration (fixtures/rag_answer_config.json) is what the code uses by default."""
import json

from backend import answer, config, conversations

def frozen():
    return json.loads((config.ROOT / "fixtures" / "rag_answer_config.json").read_text())

def test_answer_defaults_match_the_frozen_configuration():
    assert answer.DEFAULTS == {**answer.BASE_DEFAULTS, **frozen()["answer"]}

def test_rewrite_defaults_match_the_frozen_configuration():
    assert conversations.DEFAULTS == {**conversations.BASE_DEFAULTS, **frozen()["rewrite"]}

def test_default_answer_model_is_unchanged():
    assert frozen()["model"] == config.GRAPH_LLM