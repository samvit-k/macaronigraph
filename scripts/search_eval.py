"""Phase 3 retrieval evaluation: compare embedding models on hand-written queries over the real canonical
records. Read-only: it reads SQLite and embeds into a throw-away in-memory matrix per model (the production
Chroma index is not touched, and no Reels are processed). Ranking matches the API: cosine similarity,
fragment down-weighting, grouped by Reel (best chunk score).

Metrics:
  Reel level (all scored queries)
    Hit@1 / Hit@5   at least one relevant Reel ranked 1st / in the top 5
    MRR             reciprocal rank of the first relevant Reel
  OCR passage level (ocr_only queries)
    OCR Hit@1/@5    individual OCR-derived chunks ranked on their own (not grouped by Reel; captions and
                    transcripts excluded); a hit is a chunk of a relevant Reel whose text contains every
                    `ocr_terms` entry (case-insensitive). Finding the Reel via speech or caption never counts.
Validation: relevant Reel IDs are checked against the corpus (missing IDs reported; queries with none
available excluded); ocr_only queries without non-empty `ocr_terms` are excluded.

  .venv/bin/python scripts/search_eval.py [model ...]     (default: Qwen3-Embedding-0.6B vs bge-base-en-v1.5)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import chunking, config, embedder, indexer  # noqa: E402

QUERIES = ROOT / "fixtures" / "search_queries.json"
DEFAULT_MODELS = ["Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-base-en-v1.5"]
OCR_TYPES = ("ocr", "ocr_title", "ocr_fragment")

def corpus(source: str = "instagram") -> list[dict]:
    chunks = []
    for rid, e in indexer.eligible().items():
        if e["source"] == source:
            chunks += chunking.build_chunks(indexer.load_record(e["aid"]), e["source"])
    return chunks

def _valid_terms(terms) -> bool:
    return isinstance(terms, list) and bool(terms) and all(isinstance(t, str) and t.strip() for t in terms)

def validate(queries: list[dict], reels: set[str]) -> tuple[list[dict], list[dict], list[str]]:
    """→ (scored queries with `relevant` narrowed to available Reels, excluded queries + reason, warnings)"""
    scored, excluded, warnings = [], [], []
    for q in queries:
        missing = [r for r in q["relevant"] if r not in reels]
        if missing:
            warnings.append(f"{q['q']!r}: relevant Reel(s) not in corpus: {missing}")
        available = [r for r in q["relevant"] if r in reels]
        if not available:
            excluded.append({**q, "reason": "no relevant Reel in the corpus"})
        elif q["kind"] == "ocr_only" and not _valid_terms(q.get("ocr_terms")):
            excluded.append({**q, "reason": "ocr_only query without non-empty ocr_terms"})
            warnings.append(f"{q['q']!r}: excluded (ocr_only needs non-empty ocr_terms)")
        else:
            scored.append({**q, "relevant": available, "missing": missing})
    return scored, excluded, warnings

def _weighted(s: float, c: dict) -> float:
    return float(s) * (config.FRAGMENT_WEIGHT if "fragment" in c["metadata"].get("ocr_quality", "") else 1.0)

def rank_reels(sims, chunks) -> list[tuple[str, float, dict]]:
    best: dict[str, tuple[float, dict]] = {}
    for c, s in zip(chunks, sims):
        sc, rid = _weighted(s, c), c["metadata"]["reel_id"]
        if rid not in best or sc > best[rid][0]:
            best[rid] = (sc, c)
    return sorted(((r, s, c) for r, (s, c) in best.items()), key=lambda x: -x[1])

def rank_ocr_passages(sims, chunks) -> list[tuple[float, dict]]:
    """Every OCR-derived chunk on its own, best first."""
    return sorted(((_weighted(s, c), c) for c, s in zip(chunks, sims) if c["metadata"]["content_type"] in OCR_TYPES),
                  key=lambda x: -x[0])

def contains_terms(text: str, terms: list[str]) -> bool:
    return _valid_terms(terms) and all(t.lower() in text.lower() for t in terms)

def ocr_passage_rank(ranked_passages, relevant: list[str], terms: list[str]) -> tuple[int | None, dict | None]:
    """1-based rank of the first OCR passage that belongs to a relevant Reel and contains all terms."""
    for i, (_, c) in enumerate(ranked_passages):
        if c["metadata"]["reel_id"] in relevant and contains_terms(c["text"], terms):
            return i + 1, c
    return None, None

def reel_rank(ranked_reels, relevant: list[str]) -> int | None:
    return next((i + 1 for i, (r, _, _) in enumerate(ranked_reels) if r in relevant), None)

def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    return {"n": len(ranks), "hit@1": sum(1 for r in ranks if r == 1) / n,
            "hit@5": sum(1 for r in ranks if r and r <= 5) / n, "mrr": sum(1 / r for r in ranks if r) / n}

def evaluate(model: str, chunks: list[dict], queries: list[dict]) -> dict:
    emb = embedder.Embedder(embedder.current_spec(model))
    t0 = time.time()
    emb.dim                                                     # load the model
    load_s = time.time() - t0
    t0 = time.time()
    vecs = emb.encode_docs([chunking.embed_text(c) for c in chunks])
    embed_s = time.time() - t0
    per = []
    for item in queries:
        sims = vecs @ emb.encode_query(item["q"])
        ranked = rank_reels(sims, chunks)
        row = {**item, "rank": reel_rank(ranked, item["relevant"]),
               "top": [(r, round(s, 3), c["metadata"]["content_type"]) for r, s, c in ranked[:3]]}
        if item["kind"] == "ocr_only":
            r, c = ocr_passage_rank(rank_ocr_passages(sims, chunks), item["relevant"], item["ocr_terms"])
            row["ocr_rank"], row["ocr_passage"] = r, (c["id"], " ".join(c["text"].split())[:140]) if c else None
        per.append(row)
    kinds = sorted({r["kind"] for r in per})
    ocr_rows = [r for r in per if r["kind"] == "ocr_only"]
    return {"model": model, "chunks": len(chunks), "load_s": round(load_s, 1), "embed_s": round(embed_s, 1),
            "chunks_per_s": round(len(chunks) / embed_s, 1), "reel_level": metrics([r["rank"] for r in per]),
            "by_kind": {k: metrics([r["rank"] for r in per if r["kind"] == k]) for k in kinds},
            "ocr_passage_level": metrics([r["ocr_rank"] for r in ocr_rows]) if ocr_rows else None,
            "per_query": per}

def main(models: list[str]) -> int:
    if not QUERIES.is_file():
        sys.exit(f"{QUERIES.relative_to(ROOT)} not found: this private evaluation dataset is not part of the repository "
                 "(see README, Evaluation)")
    queries = json.loads(QUERIES.read_text())["queries"]
    chunks = corpus()
    reels = {c["metadata"]["reel_id"] for c in chunks}
    scored, excluded, warnings = validate(queries, reels)
    print(f"corpus: {len(chunks)} chunks from {len(reels)} Reels")
    print(f"queries: {len(queries)} total, {len(scored)} scored, {len(excluded)} excluded")
    for w in warnings:
        print(f"  ! {w}")
    results = [evaluate(m, chunks, scored) for m in models]

    print("\nReel level")
    print(f"  {'model':<24} {'Hit@1':>6} {'Hit@5':>6} {'MRR':>5}   chunks/s")
    for r in results:
        m = r["reel_level"]
        print(f"  {r['model'].split('/')[-1]:<24} {m['hit@1']:6.2f} {m['hit@5']:6.2f} {m['mrr']:5.2f}   "
              f"{r['chunks_per_s']} (load {r['load_s']} s, embed {r['embed_s']} s)")
    print("\nReel level by query category (Hit@1 / Hit@5 / MRR)")
    for k in results[0]["by_kind"]:
        cells = "   ".join(f"{r['model'].split('/')[-1]}: {r['by_kind'][k]['hit@1']:.2f}/{r['by_kind'][k]['hit@5']:.2f}/"
                           f"{r['by_kind'][k]['mrr']:.2f}" for r in results)
        print(f"  {k:<11} n={results[0]['by_kind'][k]['n']:<3} {cells}")
    if results[0]["ocr_passage_level"]:
        print("\nOCR passage level (ocr_only; OCR chunks ranked individually, must contain ocr_terms)")
        for r in results:
            o = r["ocr_passage_level"]
            print(f"  {r['model'].split('/')[-1]:<24} n={o['n']}  Hit@1 {o['hit@1']:.2f}  Hit@5 {o['hit@5']:.2f}  MRR {o['mrr']:.2f}")
        for i, item in enumerate(scored):
            if item["kind"] == "ocr_only":
                print(f"    {item['q']!r}  terms={item['ocr_terms']}  ranks={[r['per_query'][i]['ocr_rank'] for r in results]}")

    print("\nQueries not ranked 1st (Reel level) by every model:")
    for i, item in enumerate(scored):
        ranks = [r["per_query"][i]["rank"] for r in results]
        if any(x != 1 for x in ranks):
            print(f"  [{item['kind']}] {item['q']!r}  expected {item['relevant']}  ranks={ranks}")
            for r in results:
                print(f"      {r['model'].split('/')[-1]:<24} top3={r['per_query'][i]['top']}")
    out = ROOT / "output" / "search_eval.json"
    out.write_text(json.dumps({"total": len(queries), "scored": len(scored), "excluded": excluded,
                               "warnings": warnings, "results": results}, indent=1, ensure_ascii=False))
    print(f"\nfull results → {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_MODELS))