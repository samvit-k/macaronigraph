"""Phase 4B evaluation (fixtures/rag_questions.json, frozen). Read-only w.r.t. production data: it reads the
canonical records, the active Chroma collection and Neo4j; evaluation conversations go to a scratch database
(never output/conversations.db); nothing is written to pipeline.db, Chroma or Neo4j.

  .venv/bin/python scripts/rag_eval.py retrieval --split dev [--params '{"min_specificity": 0.7}'] [--out f.json]
  .venv/bin/python scripts/rag_eval.py tune --split dev --out tune.json            (grid over a few parameters)
  .venv/bin/python scripts/rag_eval.py retrieval --split heldout --frozen PARAMS.json --out f.json
  .venv/bin/python scripts/rag_eval.py answers --split heldout --frozen PARAMS.json --out f.json   (local LLM)
  .venv/bin/python scripts/rag_eval.py baseline [--params ...]          (Phase 3 queries in every mode)

Scoring rules are the ones stated in the dataset's "scoring" section. Held-out runs require --frozen (the tuned
parameters) and refuse to run if the dataset changed (sha256 check).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import conversations, indexer, rag  # noqa: E402

DATASET = ROOT / "fixtures" / "rag_questions.json"
FROZEN_SHA256 = "0c2950f727744fead718ed0a13516f850c981a7fb9425c02c3032705c1b2bef4"
SEARCH_QUERIES = ROOT / "fixtures" / "search_queries.json"
MODES = ("vector", "graph", "hybrid")
ABLATIONS = {"hybrid-no-comention": {"co_mention": False}, "hybrid-no-relations": {"relations": False}}
GRID = {"min_specificity": [0.55, 0.65, 0.75], "rel_frac": [0.5, 0.6, 0.7], "protect_top": [2, 3],
        "w_co_mention": [0.4, 0.6, 1.0], "reserve_graph": [2, 4]}


def load(split: str | None):
    raw = DATASET.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != FROZEN_SHA256:
        sys.exit(f"dataset changed (sha256 {sha[:12]} ≠ frozen {FROZEN_SHA256[:12]}): refusing to evaluate")
    qs = json.loads(raw)["questions"]
    return [q for q in qs if split in (None, q["split"])], sha


def items(qs, include_multi=False):
    """Single-question items; multi-turn first turns only unless include_multi (needs real citations)."""
    for q in qs:
        if "turns" in q:
            if include_multi:
                yield q
            else:
                t = q["turns"][0]
                yield {"id": q["id"] + ".t1", "category": "multi_turn", "split": q["split"], **t}
        else:
            yield q


# ---------------------------------------------------------------- retrieval metrics

def _rank_metrics(reels: list[str], relevant: set[str]) -> dict:
    out = {}
    for k in (5, 10):
        top = reels[:k]
        out[f"recall@{k}"] = len(relevant & set(top)) / len(relevant)
        out[f"hit@{k}"] = float(bool(relevant & set(top)))
    rr = next((1 / (i + 1) for i, r in enumerate(reels[:20]) if r in relevant), 0.0)
    out["mrr"] = rr
    return out


def score_item(item, res: dict, vector_pack: set[str] | None) -> dict:
    rel, acc = set(item.get("relevant", [])), set(item.get("acceptable", []))
    traps = set(item.get("traps", []))
    pack = res["sources"]
    pack_reels = {s["reel_id"] for s in pack}
    out = {"id": item["id"], "category": item["category"], "expect": item.get("expect"),
           "mode_used": res["mode_used"], "fallback": bool(res["fallback"]), "top": res["reels"][:10],
           "pack_reels": sorted(pack_reels), "retrieval_s": res["timing"].get("total_s")}
    if rel and item.get("expect") != "abstain":
        out.update(_rank_metrics(res["reels"], rel))
        out["pack_coverage"] = len(rel & pack_reels) / len(rel)
    else:
        out["trap_reels_in_pack"] = sorted(traps & pack_reels)
    if vector_pack is not None:
        added = [s for s in pack if s["chunk_id"] not in vector_pack]
        out["added"] = len(added)
        out["added_relevant"] = sum(1 for s in added if s["reel_id"] in rel | acc)
        out["added_detail"] = [{"reel_id": s["reel_id"], "chunk_id": s["chunk_id"],
                                "channels": [c["type"] for c in s["channels"]],
                                "relevant": s["reel_id"] in rel | acc} for s in added]
    return out


def summarize(rows: list[dict]) -> dict:
    def mean(key, sub):
        vals = [r[key] for r in sub if key in r]
        return (round(statistics.mean(vals), 3), len(vals)) if vals else (None, 0)
    cats = sorted({r["category"] for r in rows})
    out = {}
    for c in ["ALL", *cats]:
        sub = rows if c == "ALL" else [r for r in rows if r["category"] == c]
        s = {k: mean(k, sub)[0] for k in ("recall@5", "recall@10", "hit@5", "hit@10", "mrr", "pack_coverage")}
        s["n_scored"] = mean("mrr", sub)[1]
        added = sum(r.get("added", 0) for r in sub)
        s["added_passages"] = added
        s["added_precision"] = round(sum(r.get("added_relevant", 0) for r in sub) / added, 3) if added else None
        traps = [r for r in sub if "trap_reels_in_pack" in r]
        if traps:
            s["abstain_items_with_trap_in_pack"] = sum(1 for r in traps if r["trap_reels_in_pack"])
        out[c] = s
    lat = [r["retrieval_s"] for r in rows if r.get("retrieval_s") is not None]
    out["latency_s"] = {"median": round(statistics.median(lat), 3), "max": round(max(lat), 3)} if lat else None
    return out


def run_retrieval(qs, params: dict, modes=MODES, ablations=True) -> dict:
    results = {}
    vec_packs = {}
    configs = [(m, m, {}) for m in modes]
    if ablations and "hybrid" in modes:
        configs += [(name, "hybrid", opt) for name, opt in ABLATIONS.items()]
    for name, mode, opt in configs:
        rows = []
        for item in items(qs):
            res = rag.retrieve(item["q"], mode=mode, options={**params, **opt})
            if name == "vector":
                vec_packs[item["id"]] = {s["chunk_id"] for s in res["sources"]}
            rows.append(score_item(item, res, vec_packs.get(item["id"]) if name != "vector" else None))
        results[name] = {"summary": summarize(rows), "items": rows}
    return results


# ---------------------------------------------------------------- tuning (dev only)

def objective(res: dict) -> tuple:
    """Pre-registered: maximize dev hybrid mean recall@10, then pack coverage, then added-passage precision;
    configurations whose hybrid hit@5 falls below vector hit@5 are excluded (vector strength preserved)."""
    h, v = res["hybrid"]["summary"]["ALL"], res["vector"]["summary"]["ALL"]
    if (h["hit@5"] or 0) < (v["hit@5"] or 0):
        return (-1,)
    return (h["recall@10"] or 0, h["pack_coverage"] or 0, h["added_precision"] or 0)


def tune(qs) -> dict:
    keys = list(GRID)
    trials = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, values))
        res = run_retrieval(qs, params, modes=("vector", "hybrid"), ablations=False)
        trials.append({"params": params, "objective": objective(res), "hybrid": res["hybrid"]["summary"]["ALL"],
                       "vector": res["vector"]["summary"]["ALL"]})
    trials.sort(key=lambda t: t["objective"], reverse=True)
    return {"grid": GRID, "best": trials[0], "top5": trials[:5], "n_trials": len(trials)}


# ---------------------------------------------------------------- answers (local LLM)

def run_answers(qs, params: dict, mode: str) -> dict:
    from backend import ask
    scratch = Path(tempfile.mkdtemp(prefix="rag-eval-conv-")) / "conversations.db"
    conversations.db_path = lambda: scratch                      # never the user's conversation history
    rows = []
    for q in items(qs, include_multi=True):
        turns = q["turns"] if "turns" in q else [q]
        cid, prev_cited = None, []
        for i, t in enumerate(turns):
            t0 = time.perf_counter()
            r = ask.ask(t["q"], conversation_id=cid, mode=mode, options=params)
            wall = time.perf_counter() - t0
            cid = r["conversation_id"]
            rel = set(t.get("relevant", []))
            if t.get("relative_to_previous"):
                rel -= set(prev_cited)
            cited = sorted({c["reel_id"] for c in r["citations"]})
            rows.append({"id": q["id"] + (f".t{i + 1}" if "turns" in q else ""), "category": q["category"],
                         "expect": t.get("expect", q.get("expect")), "attribution": t.get("attribution", q.get("attribution", False)),
                         "question": t["q"], "standalone": r["standalone_query"], "rewrite": (r.get("rewrite") or {}).get("method"),
                         "status": r["status"], "answer": r["answer"].get("text"), "sentences": r["answer"].get("sentences"),
                         "removed": r["answer"].get("removed"), "citations": r["citations"], "cited_reels": cited,
                         "relevant": sorted(rel), "acceptable": sorted(t.get("acceptable", [])),
                         "traps": q.get("traps", []), "facts": t.get("facts", q.get("facts", [])),
                         "sources": [{"sid": s["sid"], "reel_id": s["reel_id"], "text": s["text"]} for s in r["sources"]],
                         "wall_s": round(wall, 2), "generation": r["timing"]["generation"]})
            prev_cited = cited
            print(f"{rows[-1]['id']:8} {r['status']:22} cited={cited} {wall:.1f}s", flush=True)
    return {"items": rows, "auto": answer_auto_metrics(rows),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)}


def answer_auto_metrics(rows: list[dict]) -> dict:
    """Automatic answer metrics; faithfulness is judged manually (saved separately)."""
    ans = [r for r in rows if r["expect"] in ("answer", "qualify", "disambiguate")]
    abst = [r for r in rows if r["expect"] == "abstain"]
    cites = [(r, c) for r in rows for c in r["citations"]]
    pack_ok = sum(1 for r, c in cites if any(s["sid"] == c["sid"] and s["reel_id"] == c["reel_id"] for s in r["sources"]))
    relevant = sum(1 for r, c in cites if c["reel_id"] in set(r["relevant"]) | set(r["acceptable"]))
    weak = sum(1 for r in rows for s in (r["sentences"] or []) if s.get("support") == "weak")
    sentences = sum(len(r["sentences"] or []) for r in rows)
    gens = [r["generation"] for r in rows if r.get("generation")]
    cold = [g["generate_s"] for g in gens if g.get("cold")]
    warm = [g["generate_s"] for g in gens if not g.get("cold")]
    return {
        "items": len(rows), "citations": len(cites),
        "citation_validity": round(pack_ok / len(cites), 3) if cites else None,
        "citation_relevance": round(relevant / len(cites), 3) if cites else None,
        "sentences": sentences, "weak_support_sentences": weak,
        "removed_sentences": sum(len(r["removed"] or []) for r in rows),
        "abstention_correct": f"{sum(1 for r in abst if r['status'] == 'insufficient_evidence')}/{len(abst)}",
        "false_abstention": f"{sum(1 for r in ans if r['status'] == 'insufficient_evidence')}/{len(ans)}",
        "trap_cited": sum(1 for r in abst if set(r["cited_reels"]) & set(r["traps"])),
        "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "generation_s": {"cold": cold, "warm_median": round(statistics.median(warm), 2) if warm else None,
                         "warm_p90": round(sorted(warm)[int(0.9 * (len(warm) - 1))], 2) if warm else None},
        "quote_exactness": "enforced by validation (quotes not found are removed and counted in removed_sentences)",
    }


# ---------------------------------------------------------------- Phase 3 baseline

def run_baseline(params: dict) -> dict:
    qs = json.loads(SEARCH_QUERIES.read_text())["queries"]
    out = {}
    for mode in MODES:
        rows = []
        for q in qs:
            res = rag.retrieve(q["q"], mode=mode, options=params)
            rows.append({"category": q["kind"], **_rank_metrics(res["reels"], set(q["relevant"]))})
        out[mode] = {k: round(statistics.mean(r[k] for r in rows), 3) for k in ("hit@5", "mrr")}
        out[mode]["hit@1"] = round(statistics.mean(float(r["mrr"] == 1.0) for r in rows), 3)
        out[mode]["n"] = len(rows)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["retrieval", "tune", "answers", "baseline"])
    ap.add_argument("--split", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--frozen")
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    busy = indexer.busy_reason()
    if busy:
        sys.exit(f"capture/processing active ({busy}); evaluate when idle")
    params = json.loads(Path(a.frozen).read_text())["params"] if a.frozen else json.loads(a.params)
    if a.split == "heldout" and not a.frozen and a.cmd != "baseline":
        sys.exit("held-out runs require --frozen PARAMS.json (parameters tuned on dev only)")
    qs, sha = load(a.split)
    t0 = time.time()
    if a.cmd == "retrieval":
        res = {"retrieval": run_retrieval(qs, params)}
    elif a.cmd == "tune":
        if a.split != "dev":
            sys.exit("tuning is only allowed on the dev split")
        res = {"tune": tune(qs)}
    elif a.cmd == "answers":
        res = {"answers": run_answers(qs, params, a.mode)}
    else:
        res = {"baseline": run_baseline(params)}
    res.update(dataset_sha256=sha, split=a.split, params={**rag.DEFAULTS, **params}, seconds=round(time.time() - t0, 1),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2))
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text)
    if a.cmd == "retrieval":
        for name, r in res["retrieval"].items():
            print(name, json.dumps(r["summary"]["ALL"]))
    elif a.cmd == "tune":
        print(json.dumps(res["tune"]["best"], indent=1))
    elif a.cmd == "baseline":
        print(json.dumps(res["baseline"], indent=1))
    else:
        print(json.dumps(res["answers"]["auto"], indent=1))


if __name__ == "__main__":
    main()
