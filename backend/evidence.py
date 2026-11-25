"""Phase 4D passage selection WITHIN Reels already in the context pack (no new Reels, no re-embedding, no change to
Reel-level retrieval). Diagnosis (reports/phase4c/REPORT.md §3): in 8 of 24 held-out failures the right Reel was
retrieved but not the passage that holds the answer ("Round 2 …", "the runtime starts the kernel").

For the top Reels of the pack, every other passage of that Reel in the ACTIVE collection (content hash equal to the
index state's) is scored against the question:
  lexical   rare question words / numbers / adjacent word pairs found in the passage, weighted by how few of the
            Reel's passages contain them (a word in every passage of the Reel says nothing);
  semantic  cosine between the question and the passage, from the embeddings already stored in Chroma.
A passage is added only if it passes BOTH a lexical minimum and a semantic floor relative to the pack, at most
`per_reel` per Reel and `max_add` in total, within a character budget, and not a near-duplicate of a pack passage.
Added passages are appended after the existing sources (S{n+1}…) with channel "same_reel_passage".
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import embedder, indexer

DEFAULTS = {"reels": 4, "per_reel": 2, "max_add": 4, "extra_chars": 3000, "min_lex": 0.34, "sim_floor": 0.75,
            "near_duplicate": 0.8, "min_words": 4}
_STOP = frozenset("""a an the and or but if then than that this these those there their they them it its is are was
were be been being to of in on at by for from with about as into over under also just only very more most much many
some any each every not no yes can could would should will may might must do does did done have has had having you
your yours we our us i me my he she his her what which who whom whose when where why how all both either neither
reel reels video videos says said say according creator mention mentions mentioned saved tell tells""".split())

def _tok(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9+#.\-]*[a-z0-9+#]|[a-z0-9]", s.lower()) if w not in _STOP]

def _stem(w: str) -> str:
    return w[:6] if w.isalpha() else w

def _anchors(question: str) -> tuple[list[str], list[tuple[str, str]]]:
    words = [_stem(w) for w in _tok(question) if len(w) > 2 or w.isdigit()]
    raw = re.findall(r"[a-z0-9]+", question.lower())
    pairs = [(_stem(a), _stem(b)) for a, b in zip(raw, raw[1:]) if a not in _STOP or b.isdigit()]
    pairs = [(a, b) for a, b in pairs if (a not in _STOP and b not in _STOP) or b.isdigit()]
    return list(dict.fromkeys(words)), list(dict.fromkeys(pairs))

def _lex(anchors, pairs, text: str, df: dict, n: int) -> float:
    toks = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]
    have, bigr = set(toks), set(zip(toks, toks[1:]))
    if not anchors:
        return 0.0
    idf = {a: math.log(1 + n / (1 + df.get(a, 0))) for a in anchors}
    total = sum(idf.values()) or 1.0
    score = sum(idf[a] for a in anchors if a in have) / total
    score += 0.5 * sum(1 for p in pairs if p in bigr) / max(1, len(pairs))
    return score

def _cos(q, v) -> float:
    v = np.asarray(v, dtype=np.float32)
    return float(np.dot(q, v) / ((np.linalg.norm(q) * np.linalg.norm(v)) or 1.0))

def select(question: str, retrieval: dict, options: dict | None = None) -> dict:
    """→ the retrieval dict with answer-bearing passages of already-retrieved Reels appended (in place), and
    retrieval["passage_select"] = what was added and why. Never removes or reorders existing sources."""
    p = {**DEFAULTS, **(options or {})}
    sources = retrieval.get("sources") or []
    info = {"params": p, "added": [], "considered": 0}
    retrieval["passage_select"] = info
    if not sources:
        return retrieval
    reels = list(dict.fromkeys(s["reel_id"] for s in sources))[:p["reels"]]
    in_pack = {s["chunk_id"] for s in sources}
    with indexer.reading() as name:
        if not name:
            return retrieval
        col = indexer.get_collection(name)
        got = col.get(where={"reel_id": {"$in": reels}}, include=["documents", "metadatas", "embeddings"])
        state = indexer.state_rows(name)
        qv = embedder.for_spec(indexer.spec_of(col)).encode_query(question)
    embs = got["embeddings"] if got["embeddings"] is not None else [None] * len(got["ids"])
    by_reel: dict[str, list] = {}
    for cid, doc, md, e in zip(got["ids"], got["documents"], got["metadatas"], embs):
        md = md or {}
        rid = md.get("reel_id")
        st = state.get(rid) or {}
        if st.get("status") != "indexed" or md.get("content_hash") != st.get("content_hash") or e is None:
            continue
        by_reel.setdefault(rid, []).append((cid, doc or "", md, e))
    anchors, pairs = _anchors(question)
    pack_sims = [s["similarity"] for s in sources if s.get("similarity") is not None]
    ref_sim = max(pack_sims) if pack_sims else 0.0
    pack_toks = {}
    for s in sources:
        pack_toks.setdefault(s["reel_id"], []).append(set(re.findall(r"[a-z0-9]+", str(s["text"]).lower())))
    cands = []
    for rid in reels:
        chunks = by_reel.get(rid, [])
        df: dict[str, int] = {}
        for _, doc, _, _ in chunks:
            for a in set(_stem(w) for w in re.findall(r"[a-z0-9]+", doc.lower())):
                df[a] = df.get(a, 0) + 1
        for cid, doc, md, e in chunks:
            if cid in in_pack or len(re.findall(r"[A-Za-z]{3,}", doc)) < p["min_words"]:
                continue
            info["considered"] += 1
            lex = _lex(anchors, pairs, doc, df, len(chunks))
            sim = _cos(qv, e)
            if lex < p["min_lex"] or sim < p["sim_floor"] * ref_sim:
                continue
            toks = set(re.findall(r"[a-z0-9]+", doc.lower()))
            if any(len(toks & o) / max(1, len(toks | o)) >= p["near_duplicate"] for o in pack_toks.get(rid, [])):
                continue
            cands.append((lex + sim, lex, sim, rid, cid, doc, md))
    cands.sort(key=lambda x: -x[0])
    per: dict[str, int] = {}
    used = 0
    for score, lex, sim, rid, cid, doc, md in cands:
        if len(info["added"]) >= p["max_add"] or per.get(rid, 0) >= p["per_reel"] or used + len(doc) > p["extra_chars"]:
            continue
        toks = set(re.findall(r"[a-z0-9]+", doc.lower()))
        if any(len(toks & o) / max(1, len(toks | o)) >= p["near_duplicate"] for o in pack_toks.get(rid, [])):
            continue
        per[rid] = per.get(rid, 0) + 1
        used += len(doc)
        pack_toks.setdefault(rid, []).append(toks)
        sid = f"S{len(sources) + 1}"
        sources.append({"sid": sid, "chunk_id": cid, "reel_id": rid,
                        "source_url": md.get("source_url") or f"https://www.instagram.com/reel/{rid}/",
                        "owner": md.get("owner_username"), "content_type": md.get("content_type"),
                        "start": md.get("start"), "end": md.get("end"), "text": doc, "similarity": round(sim, 4),
                        "score": None, "channels": [{"type": "same_reel_passage", "lexical": round(lex, 3),
                                                     "similarity": round(sim, 4)}]})
        info["added"].append({"sid": sid, "chunk_id": cid, "reel_id": rid, "lexical": round(lex, 3),
                              "similarity": round(sim, 4)})
    retrieval["sources"] = sources
    return retrieval