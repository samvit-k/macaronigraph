"""Phase 4C answer evaluation. Read-only w.r.t. production data (reads canonical records, the active Chroma
collection and Neo4j; evaluation conversations go to a scratch database).

  .venv/bin/python scripts/answer_eval.py packs --set v2dev --out reports/phase4c/packs_v2dev.json
  .venv/bin/python scripts/answer_eval.py run --set v2dev --config E0 --packs reports/phase4c/packs_v2dev.json \
        --out reports/phase4c/experiments/E0_v2dev.json
  .venv/bin/python scripts/answer_eval.py run --set v2heldout --final ...      (only after the set is approved)

Sets: v1dev (fixtures/rag_questions.json, dev split — frozen), v2dev, v2heldout (sealed: refused unless --final AND
fixtures/rag_questions_v2_heldout.seal.json records your approval and the file's current sha256).
Single-turn items are generated over CACHED evidence packs (identical for every configuration and model);
multi-turn items run end to end (rewriting changes retrieval), in a scratch conversation database.
Retrieval parameters are the frozen Phase 4B defaults (rag.DEFAULTS == fixtures/rag_params.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import answer, config, conversations, models, rag  # noqa: E402

SETS = {"v1dev": (ROOT / "fixtures" / "rag_questions.json", "dev"),
        "v2dev": (ROOT / "fixtures" / "rag_questions_v2_dev.json", None),
        "v2heldout": (ROOT / "fixtures" / "rag_questions_v2_heldout.json", None),
        "v3heldout": (ROOT / "fixtures" / "rag_questions_v3_heldout.json", None)}
SEAL = ROOT / "fixtures" / "rag_questions_v2_heldout.seal.json"
SEAL3 = ROOT / "fixtures" / "rag_questions_v3_heldout.seal.json"
V1_SHA = "0c2950f727744fead718ed0a13516f850c981a7fb9425c02c3032705c1b2bef4"

# Experiment configurations (answer = answer.DEFAULTS overrides; rewrite = conversations.DEFAULTS overrides;
# model = answer model id). Filled in as components land; each one differs from its predecessor in ONE thing.
Q4B = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
Q8B = "mlx-community/Qwen3-8B-4bit"
_P = {"inline_citations": True}
_P2 = {**_P, "prompt": "v2"}
CONFIGS: dict[str, dict] = {
    "E0": {"answer": {}, "rewrite": {}, "model": Q4B, "about": "Phase 4B behaviour (incl. the single-quote check)"},
    "E1": {"answer": _P, "rewrite": {}, "model": Q4B, "about": "E0 + inline citation parsing"},
    "E2": {"answer": _P2, "rewrite": {}, "model": Q4B, "about": "E1 + prompt v2"},
    "E3h": {"answer": {**_P2, "sufficiency": "hard_sim"}, "rewrite": {}, "model": Q4B,
            "about": "E2 + hard similarity gate (tau 0.55, from dev signals)"},
    "E3s": {"answer": {**_P2, "sufficiency": "soft"}, "rewrite": {}, "model": Q4B,
            "about": "E2 + soft missing-term warning in the prompt"},
    "E3hs": {"answer": {**_P2, "sufficiency": "soft+hard_sim"}, "rewrite": {}, "model": Q4B,
             "about": "E2 + both"},
    "E4": {"answer": _P2, "rewrite": {"referents": "v2"}, "model": Q4B,
           "about": "E2 + rewrite v2 (sufficiency added only if E3 shows a benefit)"},
    "E6": {"answer": {}, "rewrite": {}, "model": Q8B, "about": "E0 with Qwen3-8B (model effect alone)"},
    "E2f": {"answer": {**_P2, "min_source_words": 3}, "rewrite": {}, "model": Q4B,
            "about": "E2 + OCR fragments (<3 real words) hidden from the model"},
    "E2fa": {"answer": {**_P2, "min_source_words": 3, "absence_statements": "to_missing"}, "rewrite": {}, "model": Q4B,
             "about": "E2f + absence statements become the fixed missing note"},
    "E2fah": {"answer": {**_P2, "min_source_words": 3, "absence_statements": "to_missing", "sufficiency": "hard_sim"},
              "rewrite": {}, "model": Q4B, "about": "E2fa + hard similarity gate (tau 0.55)"},
    # After the trailing-marker parser fix and the follow-up exemption for hard gates (runs named *_fix):
    "E3c": {"answer": {**_P2, "min_source_words": 3, "absence_statements": "to_missing", "sufficiency": "hard_sim_anchor"},
            "rewrite": {}, "model": Q4B, "about": "E2fa + combined gate (sim < 0.55 AND a question term in no passage)"},
    "E4c": {"answer": {**_P2, "min_source_words": 3, "absence_statements": "to_missing", "sufficiency": "hard_sim_anchor"},
            "rewrite": {"referents": "v2"}, "model": Q4B, "about": "E3c + rewrite v2"},
    "E5": {"answer": {**_P2, "min_source_words": 3, "absence_statements": "to_missing", "sufficiency": "hard_sim_anchor"},
           "rewrite": {"referents": "v2"}, "model": Q8B, "about": "E4c with Qwen3-8B (combined)"},
}
# Phase 4D: D0 = the frozen Phase 4C configuration (E4c). Every component is evaluated on top of the two bug fixes
# (Dfix); D6 combines the components that helped on dev/diagnostic data.
_E4C = CONFIGS["E4c"]
_FIX = {**_E4C["answer"], "absence_clause": True, "scoped_follow_up": True}
CONFIGS.update({
    "D0": {**_E4C, "about": "frozen Phase 4C configuration (E4c)"},
    "Dfix": {"answer": _FIX, "rewrite": {"referents": "v2"}, "model": Q4B,
             "about": "D0 + the two bug fixes (absence clause; 'which of them' follow-ups exempt from hard gates)"},
    "D1": {"answer": {**_FIX, "listing": True}, "rewrite": {"referents": "v2"}, "model": Q4B,
           "about": "Dfix + deterministic listings/counts"},
    "DPS": {"answer": {**_FIX, "passage_select": True}, "rewrite": {"referents": "v2"}, "model": Q4B,
            "about": "Dfix + passage selection within retrieved Reels"},
    "D2": {"answer": {**_FIX, "verify": "drop"}, "rewrite": {"referents": "v2"}, "model": Q4B,
           "about": "Dfix + claim verification (problem claims removed)"},
    "D3": {"answer": {**_FIX, "verify": "repair"}, "rewrite": {"referents": "v2"}, "model": Q4B,
           "about": "Dfix + verification with one bounded repair"},
    "D4": {"answer": {**_FIX, "verify": "repair", "coverage": True}, "rewrite": {"referents": "v2"}, "model": Q4B,
           "about": "D3 + multi-part coverage"},
    "D5": {"answer": _FIX, "rewrite": {"referents": "v2", "clarify": True}, "model": Q4B,
           "about": "Dfix + clarification of ambiguous follow-ups"},
    "D6": {"answer": {**_FIX, "listing": True, "passage_select": True}, "rewrite": {"referents": "v2", "clarify": True},
           "model": Q4B, "about": "combined: fixes + deterministic listings + passage selection + clarification"},
    "D6v": {"answer": {**_FIX, "listing": True, "passage_select": True, "verify": "drop"},
            "rewrite": {"referents": "v2", "clarify": True}, "model": Q4B, "about": "D6 + claim verification (drop)"},
})

# ---------------------------------------------------------------- data

def load(name: str, final: bool = False):
    path, split = SETS[name]
    if not path.is_file():
        sys.exit(f"{path.relative_to(ROOT)} not found: this private evaluation dataset is not part of the repository "
                 "(see README, Evaluation)")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if name == "v1dev" and sha != V1_SHA:
        sys.exit("v1 dataset changed: refusing")
    if name == "v2heldout":            # Phase 4C held-out, now DIAGNOSTIC data (Phase 4D); still integrity-checked
        seal = json.loads(SEAL.read_text()) if SEAL.exists() else {}
        if not seal.get("approved_by_user") or seal.get("sha256") != sha:
            sys.exit("v2heldout changed since its approval: refusing")
    if name == "v3heldout":            # Phase 4D held-out: sealed
        seal = json.loads(SEAL3.read_text()) if SEAL3.exists() else {}
        if not final:
            sys.exit("v3heldout is sealed: pass --final (only after approval)")
        if not seal.get("approved_by_user") or seal.get("sha256") != sha:
            sys.exit("v3heldout is not approved, or changed since approval: refusing")
    qs = json.loads(raw)["questions"]
    return [q for q in qs if split in (None, q.get("split"))], sha

def turns_of(q):
    return q["turns"] if "turns" in q else [q]

# ---------------------------------------------------------------- configuration

def apply(cfg: dict) -> None:
    answer.DEFAULTS.clear(); answer.DEFAULTS.update({**answer.BASE_DEFAULTS, **cfg.get("answer", {})})
    conversations.DEFAULTS.clear(); conversations.DEFAULTS.update({**conversations.BASE_DEFAULTS, **cfg.get("rewrite", {})})
    config.RAG_LLM = cfg.get("model", config.RAG_LLM)

# ---------------------------------------------------------------- memory monitor

class Monitor:
    """Samples macOS memory-pressure level (kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical)
    and swap-outs (vm_stat) once a second."""

    def __init__(self):
        self.levels, self._stop = [], threading.Event()
        self.swapouts0 = self._swapouts()
        self.t = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _sysctl(name):
        try:
            return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip())
        except (ValueError, OSError):
            return None

    @staticmethod
    def _swapouts():
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            line = next(x for x in out.splitlines() if x.startswith("Swapouts"))
            return int(line.split(":")[1].strip().rstrip("."))
        except (StopIteration, ValueError, OSError):
            return None

    def _run(self):
        while not self._stop.wait(1.0):
            v = self._sysctl("kern.memorystatus_vm_pressure_level")
            if v is not None:
                self.levels.append(v)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join(3)

    def report(self):
        s1 = self._swapouts()
        return {"pressure_level_max": max(self.levels) if self.levels else None,
                "pressure_samples": len(self.levels),
                "pressure_warn_or_worse_s": sum(1 for x in self.levels if x >= 2),
                "swapouts_delta": (s1 - self.swapouts0) if None not in (s1, self.swapouts0) else None,
                "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)}

# ---------------------------------------------------------------- packs

def build_packs(qs) -> dict:  # Phase 4C/4D packs: frozen retrieval, no passage selection (applied per run)
    out = {}
    for q in qs:
        if "turns" in q:
            continue
        r = rag.retrieve(q["q"], mode="hybrid")
        out[q["id"]] = r
    return out

# ---------------------------------------------------------------- run

def _row(q, t, tid, question, standalone, rewrite, retrieval, out, wall):
    return {"id": tid, "category": q["category"], "expect": t.get("expect", "answer"), "question": question,
            "standalone": standalone, "rewrite": rewrite, "status": out.get("status"), "answer": out.get("text"),
            "sentences": out.get("sentences") or [], "removed": out.get("removed") or [],
            "citations": out.get("citations") or [], "cited_reels": sorted({c["reel_id"] for c in out.get("citations") or []}),
            "relevant": t.get("relevant", []), "acceptable": t.get("acceptable", []), "traps": q.get("traps", []),
            "points": t.get("points", []), "must_not": t.get("must_not", []),
            "topic_change": t.get("topic_change", False), "referent": t.get("referent"),
            "relative_to_previous": t.get("relative_to_previous", False), "scope_previous": t.get("scope_previous", False),
            "sources": [{"sid": s["sid"], "reel_id": s["reel_id"], "chunk_id": s["chunk_id"], "text": s["text"]}
                        for s in retrieval.get("sources", [])],
            "pack_reels": sorted({s["reel_id"] for s in retrieval.get("sources", [])}),
            "gate": out.get("gate"), "raw": out.get("raw"), "timing": out.get("timing"), "wall_s": round(wall, 2),
            "verification": out.get("verification"), "deterministic": out.get("deterministic"),
            "passage_select": retrieval.get("passage_select"), "listing": t.get("listing"),
            "clarify": out.get("clarify")}

def run(qs, packs: dict) -> list[dict]:
    from backend import ask
    scratch = Path(tempfile.mkdtemp(prefix="answer-eval-conv-")) / "conversations.db"
    conversations.db_path = lambda: scratch                      # never the user's conversation history
    rows = []
    for q in qs:
        if "turns" not in q:
            import copy
            from backend import evidence, listing
            retrieval = copy.deepcopy(packs[q["id"]])
            t0 = time.perf_counter()
            if answer.DEFAULTS.get("passage_select") and not (answer.DEFAULTS.get("listing") and listing.detect(q["q"])):
                evidence.select(retrieval["question"], retrieval)
            out = answer.respond(q["q"], retrieval)
            rows.append(_row(q, q, q["id"], q["q"], q["q"], "none", retrieval, out, time.perf_counter() - t0))
        else:
            cid, prev_cited = None, []
            for i, t in enumerate(q["turns"]):
                t0 = time.perf_counter()
                r = ask.ask(t["q"], conversation_id=cid, mode="hybrid")
                wall = time.perf_counter() - t0
                cid = r["conversation_id"]
                turn = conversations.get_turn(r["turn_id"])
                out = {**(turn.get("answer") or {}), "status": r["status"]}
                tt = dict(t)
                if t.get("relative_to_previous"):
                    tt["relevant"] = [x for x in t["relevant"] if x not in prev_cited]
                row = _row(q, tt, f"{q['id']}.t{i + 1}", t["q"], r["standalone_query"],
                           (r.get("rewrite") or {}).get("method"), turn["retrieval"], out, wall)
                row["timing"] = r["timing"].get("generation")
                rows.append(row)
                prev_cited = row["cited_reels"]
        print(f"{rows[-1]['id']:8} {rows[-1]['status']:22} cited={rows[-1]['cited_reels']} {rows[-1]['wall_s']}s", flush=True)
    return rows

# ---------------------------------------------------------------- automatic metrics (proxies)

def _hit(p, text):
    return any(k in text for k in p.get("kw", []))

def auto_metrics(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["expect"] in ("answer", "qualify", "disambiguate")]
    abst = [r for r in rows if r["expect"] == "abstain"]
    cites = [(r, c) for r in rows for c in r["citations"]]
    valid = sum(1 for r, c in cites if any(s["sid"] == c["sid"] and s["chunk_id"] == c["chunk_id"] for s in r["sources"]))
    relevant = sum(1 for r, c in cites if c["reel_id"] in set(r["relevant"]) | set(r["acceptable"]))
    pts = [(r, p) for r in ans for p in r["points"]]
    covered = sum(1 for r, p in pts if r["status"] == "ok" and _hit(p, (r["answer"] or "").lower()))
    full = sum(1 for r in ans if r["status"] == "ok" and r["points"]
               and all(_hit(p, (r["answer"] or "").lower()) for p in r["points"]))
    mn = sum(1 for r in rows if any(_hit(p, (r["answer"] or "").lower()) for p in r["must_not"]) and r["status"] == "ok")
    gens = [r["timing"] for r in rows if r.get("timing") and r["timing"].get("generate_s")]
    ans_s = sorted(r["timing"]["answer_s"] for r in rows if (r.get("timing") or {}).get("answer_s")
                   and not r["timing"].get("cold"))
    lst = [r for r in rows if r.get("listing") or (r.get("deterministic") and r["relevant"])]
    def _listing(r):
        req = set((r.get("listing") or {}).get("required") or r["relevant"])
        ok = req | set((r.get("listing") or {}).get("acceptable") or r["acceptable"])
        got = set(r["cited_reels"])
        return {"complete": req <= got, "recall": len(req & got) / len(req) if req else 1.0, "extra": sorted(got - ok)}
    lm = [_listing(r) for r in lst]
    vers = [r["verification"] for r in rows if r.get("verification")]
    cold = [g["generate_s"] for g in gens if g.get("cold")]
    warm = sorted(g["generate_s"] for g in gens if not g.get("cold"))
    removed = [x for r in rows for x in r["removed"]]
    return {
        "items": len(rows), "answerable": len(ans), "abstain_items": len(abst),
        "correct_abstention": f"{sum(1 for r in abst if r['status'] == 'insufficient_evidence')}/{len(abst)}",
        "false_abstention": f"{sum(1 for r in ans if r['status'] == 'insufficient_evidence')}/{len(ans)}",
        "trap_cited": sum(1 for r in abst if set(r["cited_reels"]) & set(r["traps"])),
        "citations": len(cites), "citation_validity": round(valid / len(cites), 3) if cites else None,
        "citation_relevance": round(relevant / len(cites), 3) if cites else None,
        "points_covered_proxy": f"{covered}/{len(pts)}", "items_all_points_proxy": f"{full}/{len(ans)}",
        "must_not_hits_proxy": mn,
        "sentences": sum(len(r["sentences"]) for r in rows), "removed_sentences": len(removed),
        "removed_by_reason": {k: sum(1 for x in removed if x["reason"].startswith(k))
                              for k in ("quote not found", "no valid citation", "inference without", "malformed")},
        "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "generation_s": {"cold": cold, "warm_median": round(statistics.median(warm), 2) if warm else None,
                         "warm_p90": round(warm[int(0.9 * (len(warm) - 1))], 2) if warm else None,
                         "n_warm": len(warm)},
        "model_load_s": [g.get("model_load_s") for g in gens if g.get("model_load_s")],
        "answer_s_warm": {"median": round(statistics.median(ans_s), 2) if ans_s else None,
                          "p90": round(ans_s[int(0.9 * (len(ans_s) - 1))], 2) if ans_s else None, "n": len(ans_s)},
        "listing": {"items": len(lm), "complete": sum(x["complete"] for x in lm),
                    "recall": round(sum(x["recall"] for x in lm) / len(lm), 3) if lm else None,
                    "extra_reels_named": sum(len(x["extra"]) for x in lm),
                    "deterministic_answers": sum(1 for r in rows if r.get("deterministic"))},
        "verification": {"answers": len(vers), "claims": sum(len(v["claims"]) for v in vers),
                         "problems": sum(1 for v in vers for c in v["claims"] if c["decision"] == "problem"),
                         "repairs_accepted": sum((v.get("repair") or {}).get("accepted", 0) for v in vers),
                         "model_calls": sum(v["calls"] for v in vers)},
        "clarify": sum(1 for r in rows if r["status"] == "clarify"),
        "passage_select_added": sum(len((r.get("passage_select") or {}).get("added") or []) for r in rows),
    }

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["packs", "run"])
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--config", default="E0")
    ap.add_argument("--packs")
    ap.add_argument("--out")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args(argv)
    qs, sha = load(a.set, final=a.final)
    meta = {"set": a.set, "dataset_sha256": sha, "rag_params": dict(rag.DEFAULTS)}
    if a.cmd == "packs":
        res = {**meta, "packs": build_packs(qs)}
    else:
        cfg = CONFIGS[a.config]
        apply(cfg)
        packs = json.loads(Path(a.packs).read_text())["packs"] if a.packs else build_packs(qs)
        if a.packs and json.loads(Path(a.packs).read_text())["dataset_sha256"] != sha:
            sys.exit("packs were built for a different dataset version")
        t0 = time.time()
        with Monitor() as mon:
            rows = run(qs, packs)
        models.unload_if_idle(0)
        res = {**meta, "config": a.config, "config_detail": cfg, "answer_defaults": dict(answer.DEFAULTS),
               "rewrite_defaults": dict(conversations.DEFAULTS), "model": config.RAG_LLM,
               "prompt_sha256": answer.prompt_sha256(), "max_tokens": answer.MAX_ANSWER_TOKENS, "temperature": 0.0,
               "seconds": round(time.time() - t0, 1), "memory": mon.report(), "auto": auto_metrics(rows), "items": rows}
        print(json.dumps({"auto": res["auto"], "memory": res["memory"]}, indent=1))
    text = json.dumps(res, indent=1, ensure_ascii=False, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")

if __name__ == "__main__":
    main()