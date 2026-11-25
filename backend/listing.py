"""Phase 4D deterministic listings: "Which Reels mention X?", "How many Reels mention X?", "Which of them also
mention X?", "Which other Reels mention X?" — answered WITHOUT the language model, from an exhaustive literal
search of the active semantic index.

What counts as a match (a literal mention, never a semantic relationship):
  - a case-insensitive, word-boundary match of the term or one of its aliases in a passage's text;
  - aliases are: spacing/hyphen variants of the term ("leet code" / "leetcode"), and, when the knowledge graph is
    reachable, the canonical name and curated aliases of the ONE graph concept the term links to. Nothing fuzzy;
  - only passages of the ACTIVE collection, of Instagram Reels, whose stored content hash equals the index state's
    hash for a Reel whose index row is current for its canonical attempt (stale or partial Reels are skipped and
    counted);
  - every listed Reel is cited with the passage that contains the mention, and the evidence type is kept apart:
    spoken transcript or caption vs on-screen text (OCR, which can be misread).
The answer states how many Reels were searched (processed and indexed) and how many saved items are not processed
yet — it never claims to cover all saved Reels. Questions with any further constraint ("…and in what context",
"…talk about", "…recommend") are not handled here (→ normal generation).
"""
from __future__ import annotations

import re

from . import indexer, jobs

_VERB = r"(?:mention|mentions|mentioned|name|names|named|reference|references|referenced)"
_REELS = r"(?:my\s+)?(?:saved\s+)?(?:other\s+)?(?:reels?|videos?)"
_PATTERNS = [
    ("count", re.compile(r"^\s*how many\s+(?:of\s+)?(?:my\s+)?(?:saved\s+)?(?P<other>other\s+)?(?:reels?|videos?)\s+"
                         r"(?:also\s+)?(?:do\s+)?" + _VERB + r"\s+(?P<tail>.+?)\s*\??\s*$", re.I)),
    ("within", re.compile(r"^\s*which (?:ones )?of (?:them|these|those)\s+(?:also\s+)?" + _VERB +
                          r"\s+(?P<tail>.+?)\s*\??\s*$", re.I)),
    ("list", re.compile(r"^\s*(?:which|what|list|show me|do any of)\s+(?:of\s+)?(?:my\s+)?(?:saved\s+)?"
                        r"(?P<other>other\s+)?(?:my\s+)?(?:saved\s+)?(?:reels?|videos?)\s+(?:also\s+)?" + _VERB +
                        r"\s+(?P<tail>.+?)\s*\??\s*$", re.I)),
]
_PRONOUN_TAIL = re.compile(r"^(?:it|this|that|them|these|those|this one|that one)$", re.I)
_BAD_TAIL = re.compile(r"\b(?:and|or|in what|how|why|when|where|which|who|context|but|with|about|for|that|"
                       r"recommend|explain|say)\b", re.I)
OCR_TYPES = {"ocr", "ocr_title", "ocr_screen", "ocr_fragment"}
TYPE_LABEL = {"transcript": "transcript", "caption": "caption"}
MAX_TERM_WORDS = 5


def detect(question: str) -> dict | None:
    """→ {kind: list|count|within, other: bool, term: str|None (None = pronoun, resolve from context)} or None."""
    q = " ".join((question or "").split())
    for kind, pat in _PATTERNS:
        m = pat.match(q)
        if not m:
            continue
        tail = m.group("tail").strip().strip("\"'“”‘’").rstrip(".?!").strip()
        if _PRONOUN_TAIL.match(tail):
            return {"kind": kind, "other": bool(m.groupdict().get("other")), "term": None}
        if not tail or len(tail.split()) > MAX_TERM_WORDS or _BAD_TAIL.search(tail):
            return None
        return {"kind": kind, "other": bool(m.groupdict().get("other")), "term": tail}
    return None


def _variants(term: str) -> set[str]:
    camel = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", term)            # "LeetCode" → "Leet Code", "GitHub" → "Git Hub"
    if camel != term:
        return _variants(camel.lower()) | _variants(term.lower())
    t = " ".join(term.lower().split())
    out = {t, t.replace("-", " "), t.replace(" ", ""), t.replace(" ", "-"), t.replace("-", "")}
    if t.endswith("'s"):
        out.add(t[:-2])
    return {v for v in out if v}


def aliases(term: str, graph=None) -> tuple[list[str], str | None]:
    """Surface forms to search for. → (forms, note). Graph aliases only when the term links to exactly one concept
    covering the whole term."""
    forms = set(_variants(term))
    note = None
    try:
        from . import rag
        g = graph or rag.GraphReader()
        index = g.concept_index()
        links = rag._link(term, index)
        words = rag._words(term)
        if len(links) == 1 and links[0][1] == 0 and links[0][2] == len(words):
            c = index["concepts"][links[0][0]]
            for s in [c["name"], *(c.get("aliases") or [])]:
                if s and len(s) >= 2:
                    forms |= _variants(s)
    except Exception as e:  # noqa: BLE001 — graph offline: literal forms of the term only
        note = f"graph aliases unavailable ({type(e).__name__})"
    return sorted(forms, key=lambda f: (-len(f), f)), note


def _pattern(forms: list[str]) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(f) for f in forms) + r")(?![A-Za-z0-9])", re.I)


def _corpus_counts() -> dict:
    with jobs.connect() as con:
        rows = con.execute("SELECT state, COUNT(*) FROM reels WHERE source='instagram' GROUP BY state").fetchall()
    return {r[0]: r[1] for r in rows}


def search(forms: list[str]) -> dict:
    """Exhaustive literal search of the active collection (read-only). → {matches: {reel_id: [hit]}, searched,
    passages, skipped_stale, collection}"""
    pat = _pattern(forms)
    with indexer.reading() as name:
        if not name:
            raise indexer.IndexNotReady("no semantic index has been built yet")
        col = indexer.get_collection(name)
        got = col.get(where={"source": "instagram"}, include=["documents", "metadatas"])
        state = indexer.state_rows(name)
        elig = indexer.eligible()
    current = {rid for rid, st in state.items()
               if st["status"] == "indexed" and rid in elig and st["canonical_attempt_id"] == elig[rid]["aid"]}
    matches: dict[str, list[dict]] = {}
    searched, stale, n = set(), set(), 0
    for cid, doc, md in zip(got["ids"], got["documents"], got["metadatas"]):
        md = md or {}
        rid = md.get("reel_id")
        if rid not in current or md.get("content_hash") != state[rid]["content_hash"]:
            stale.add(rid)
            continue
        searched.add(rid)
        n += 1
        m = pat.search(doc or "")
        if m:
            matches.setdefault(rid, []).append({"chunk_id": cid, "reel_id": rid, "text": doc,
                                                "content_type": md.get("content_type"), "start": md.get("start"),
                                                "end": md.get("end"), "owner": md.get("owner_username"),
                                                "source_url": md.get("source_url"), "match": m.group(0),
                                                "span": [m.start(), m.end()]})
    return {"matches": matches, "searched": sorted(searched), "passages": n, "skipped_stale": sorted(stale - searched),
            "collection": name}


def _rank(hit: dict) -> int:
    t = hit.get("content_type")
    return 0 if t == "transcript" else 1 if t == "caption" else 2


def _snippet(hit: dict, width: int = 70) -> str:
    a, b = hit["span"]
    t = hit["text"]
    s, e = max(0, a - width), min(len(t), b + width)
    return ("…" if s else "") + " ".join(t[s:e].split()) + ("…" if e < len(t) else "")


def answer(question: str, scope: dict | None = None, graph=None) -> tuple[dict, list[dict]] | None:
    """Deterministic answer for a listing/count question, or None if the question is not one (or the term cannot be
    resolved). → (answer object in the answer.validate shape, sources [S1..Sn] = one evidence passage per Reel).
    scope: {"within": [reel ids]} (which of them …), {"exclude": [...]} (which other …), "previous_term": str."""
    d = detect(question)
    scope = scope or {}
    if not d:
        return None
    term = d["term"] or scope.get("previous_term")
    if not term:
        return None
    forms, note = aliases(term, graph)
    res = search(forms)
    within = set(scope.get("within") or []) if d["kind"] == "within" else None
    if d["kind"] == "within" and not within:
        return None                                  # "which of them" without a previous answer: not deterministic
    exclude = set(scope.get("exclude") or []) if d["other"] else set()
    reels = [r for r in res["matches"] if (within is None or r in within) and r not in exclude]
    best = {r: sorted(res["matches"][r], key=_rank)[0] for r in reels}
    spoken = sorted((r for r in reels if _rank(best[r]) < 2), key=lambda r: (_rank(best[r]), r))
    ocr_only = sorted(r for r in reels if _rank(best[r]) == 2)
    ordered = spoken + ocr_only
    sources = []
    for i, r in enumerate(ordered):
        h = best[r]
        sources.append({"sid": f"S{i + 1}", "chunk_id": h["chunk_id"], "reel_id": r,
                        "source_url": h["source_url"] or f"https://www.instagram.com/reel/{r}/", "owner": h["owner"],
                        "content_type": h["content_type"], "start": h["start"], "end": h["end"], "text": h["text"],
                        "similarity": None, "score": None,
                        "channels": [{"type": "literal_mention", "match": h["match"]}]})
    counts = _corpus_counts()
    n_search = len(res["searched"])
    total = sum(counts.values())
    pool = (f"the {len(within)} Reels of the previous answer" if within is not None else
            f"the {n_search} processed Reels" + (" (excluding the previous answer's Reels)" if exclude else ""))
    shown = f"“{term}”"
    sentences = []
    if d["kind"] == "count" or not ordered:
        head = (f"{len(ordered)} of {pool} mention {shown}" if ordered else f"None of {pool} mention {shown}")
        if ordered:
            head += (f": {len(spoken)} in the spoken transcript or caption"
                     + (f" and {len(ocr_only)} only in on-screen text" if ocr_only else "")) if ocr_only else ""
        sentences.append({"text": head + ".", "kind": "search", "citations": [], "support": "deterministic",
                          "support_score": 1.0})
    for s in sources:
        where = TYPE_LABEL.get(s["content_type"], "on-screen text (OCR, may be misread)")
        who = f"@{s['owner']}'s Reel" if s.get("owner") else f"Reel {s['reel_id']}"
        sentences.append({"text": f"{who} mentions {shown} in its {where}: “{_snippet(best[s['reel_id']])}”",
                          "kind": "source", "citations": [s["sid"]], "support": "deterministic", "support_score": 1.0})
    scope_note = (f"Exact search of transcripts, captions and on-screen text in {n_search} processed Reels "
                  f"({res['passages']} passages); {max(0, total - n_search)} of your {total} recorded saved items "
                  "are not processed or not searchable yet and were not searched. Literal mentions only — Reels that "
                  "discuss the topic without naming it are not listed.")
    if res["skipped_stale"]:
        scope_note += f" {len(res['skipped_stale'])} Reels were skipped because their index entry is out of date."
    if note:
        scope_note += f" Only the exact wording was searched ({note})."
    cited = [{k: s.get(k) for k in ("sid", "reel_id", "source_url", "chunk_id", "content_type", "start", "end", "owner")}
             for s in sources]
    from .answer import render
    text = render([x for x in sentences], "", "ok")
    text = text.replace(" \n", "\n") + "\n\n" + scope_note
    det = {"kind": d["kind"], "term": term, "forms": forms, "reels": ordered, "spoken_or_caption": spoken,
           "ocr_only": ocr_only, "searched_reels": n_search, "searched_passages": res["passages"],
           "not_processed": max(0, total - n_search), "skipped_stale": res["skipped_stale"],
           "scope": "within_previous" if within is not None else "exclude_previous" if exclude else "all",
           "complete": not res["skipped_stale"], "note": note, "collection": res["collection"]}
    out = {"status": "ok", "sentences": sentences, "removed": [], "missing": "", "conflict": False,
           "model_said_insufficient": False, "citations": cited, "text": text, "deterministic": det,
           "generated": False, "gate": None, "timing": {"generate_s": 0.0, "model_load_s": 0.0, "cold": False}}
    return out, sources
