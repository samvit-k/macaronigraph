"""Phase 4C evidence-sufficiency signals (experimental; deterministic, no model).

signals(question, retrieval) →
  best_sim      highest vector similarity of any passage in the pack (None if no vector-ranked passage);
  anchors       the question's content words (light stemming) — not stopwords, not question/meta words;
  missing       anchors that occur in no passage of the pack;
  coverage      share of anchors found in the pack.
These are SIGNALS, not proof: GraphRAG can retrieve relevant text with low similarity or different wording, and
a word can occur in a passage that does not answer the question. answer.generate() uses them only as configured
(answer.DEFAULTS["sufficiency"]):
  "off"          no effect;
  "soft"         missing anchors are listed in the prompt as a warning (the model still decides);
  "hard_sim"     abstain without generating if best_sim < τ_sim;
  "hard_anchor"  abstain without generating if coverage < τ_cov;
  "soft+hard_sim" both;
  "hard_sim_anchor"  abstain without generating only if BOTH best_sim < τ_sim AND at least one of the question's
                 terms occurs in no passage (low similarity alone is not enough: entity lookups answered through the
                 graph or lexical channel can have low similarity — dev item "What is <tool>?" had 0.45).
Hard gates are never applied to follow-up turns (answer.generate(follow_up=True)): their signals would be computed on
a question whose meaning depends on the conversation.
Thresholds are chosen on development data only (fixtures/rag_answer_config.json).
"""
from __future__ import annotations

import re

from .graph.extract import norm_text

_STOP = frozenset("""a an the and or but if then than that this these those there their they them it its is are was
were be been being to of in on at by for from with about as into over under also just only very more most much many
some any each every not no yes can could would should will may might must do does did done have has had having you
your yours we our us i me my he she his her what which who whom whose when where why how all both either neither
reel reels video videos say says said tell tells told according creator mention mentions mentioned saved talk talks
recommend recommends recommended describe describes suggest suggests give gives explain explains show shows
other another one ones thing things get make way ways good best really need use used using like
""".split())
TAU_SIM = 0.55
TAU_COV = 0.5

def _stem(w: str) -> str:
    return w[:5] if w.isalpha() and len(w) > 5 else w

def anchor_words(question: str) -> dict[str, str]:
    """stem → the question's own word (first occurrence)."""
    words = re.findall(r"[a-z0-9][a-z0-9+#.\-]*[a-z0-9+#]|[a-z0-9]", question.lower())
    out: dict[str, str] = {}
    for w in words:
        w = w.strip(".-")
        if w in _STOP or (len(w) < 3 and not any(ch.isdigit() for ch in w)):
            continue
        out.setdefault(_stem(w), w)
    return out

def anchors(question: str) -> list[str]:
    return list(anchor_words(question))

def signals(question: str, retrieval: dict) -> dict:
    sims = [s.get("similarity") for s in retrieval.get("sources", []) if s.get("similarity") is not None]
    pool = " " + " ".join(norm_text(str(s.get("text", ""))) for s in retrieval.get("sources", [])) + " "
    words = {_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9+#.\-]*[a-z0-9+#]|[a-z0-9]", pool)}
    aw = anchor_words(question)
    anc = list(aw)
    missing = [a for a in anc if a not in words and f" {a}" not in pool]
    return {"best_sim": round(max(sims), 4) if sims else None, "anchors": anc, "missing": missing,
            "missing_words": [aw[a] for a in missing],
            "coverage": round(1 - len(missing) / len(anc), 3) if anc else 1.0}

def decide(mode: str, sig: dict, tau_sim: float = TAU_SIM, tau_cov: float = TAU_COV) -> str | None:
    """→ a reason to abstain without generating, or None."""
    if mode == "hard_sim_anchor":
        if sig["best_sim"] is not None and sig["best_sim"] < tau_sim and sig["missing"]:
            return (f"best vector similarity {sig['best_sim']:.2f} < {tau_sim} and no passage contains: "
                    f"{', '.join(sig['missing_words'])}")
        return None
    if "hard_sim" in mode and sig["best_sim"] is not None and sig["best_sim"] < tau_sim:
        return f"best vector similarity {sig['best_sim']:.2f} < {tau_sim}"
    if "hard_anchor" in mode and sig["coverage"] < tau_cov:
        return f"only {sig['coverage']:.0%} of the question's terms occur in the retrieved passages"
    return None