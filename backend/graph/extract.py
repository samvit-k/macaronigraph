"""Graph extraction: a local LLM proposes concepts and relations, and deterministic code grounds them in the passages."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher

from .. import chunking
from . import normalize as nz

PROMPT_VERSION = "p4"
INPUT_RULES_VERSION = "i1"
RECOVERY_VERSION = "r2"
RETRY_MAX_TOKENS = 3000
RETRY_HINT = ("\n\nReturn ONLY the minified JSON line, at most 12 concepts and 6 relations, each concept once, "
              "with at most 3 passage numbers per concept.")
GROUNDING_RULES = "g3"
SPOKEN_OR_CAPTION = ("transcript", "caption")
MIN_EVIDENCE_WORDS = 4
MIN_QUOTE_WORDS = 4
MAX_QUOTE_WORDS = 40
FUZZY = 0.85
FUZZY_MIN_CHARS = 5
MAX_SURFACE_WORDS = 4
MAX_CONCEPTS, MAX_RELATIONS = 15, 8
MAX_NEW_TOKENS = 1500
PREDICATES = {
    "IS_A": "X is a kind/instance of Y",
    "PART_OF": "X is a component or part of Y",
    "USES": "X uses / is built with / runs Y",
    "DEPENDS_ON": "X requires Y to work",
    "USED_FOR": "X is used for / helps with Y",
    "ALTERNATIVE_TO": "X can replace / competes with Y",
    "CONTRASTS_WITH": "X is explicitly compared or contrasted with Y (different approach / trade-off)",
    "PREREQUISITE_FOR": "learn or do X before Y",
}

SYSTEM = ("You extract a small, precise knowledge graph from ONE short video (its caption, speech transcript "
          "and on-screen text). Output a single JSON object and nothing else.")

USER_TEMPLATE = """Extract the specific things this video teaches or mentions, and relations it states explicitly.

CONCEPTS: specific technologies, tools, products, programming languages, libraries, methods/techniques,
named ideas, organizations, notable people, and resources (books, papers, GitHub repos, courses, programs).
NOT: generic words (video, tips, project, career, skills), hashtags, calls to action ("comment X"),
emotions, the narrator. Each concept is an object:
  "n": its standard short name (you may fix an obvious speech-to-text error)
  "t": one of {types}
  "s": the exact words for it as written in the passage (a name, not a sentence)
  "p": list of passage numbers where it appears

RELATIONS only when a passage explicitly states them. Predicates:
{predicates}
Each relation is an object: "s": subject concept name, "p": PREDICATE, "o": object concept name,
"i": passage number, "q": 4-20 words copied EXACTLY from that passage that state the relation.
Use only what the passages say, never general knowledge.

At most {max_c} concepts and {max_r} relations; fewer is fine. Reply with ONE line of minified JSON in this shape
(the <...> are placeholders, not content):
{{"c":[{{"n":"<name>","t":"<type>","s":"<exact words>","p":[<number>]}}],"r":[{{"s":"<name>","p":"<PREDICATE>","o":"<name>","i":<number>,"q":"<exact words>"}}]}}

PASSAGES:
{passages}
{context}"""

def _r1_version(model_id: str, revision: str) -> str:
    h = hashlib.sha256(f"{model_id}@{revision}|{PROMPT_VERSION}|{INPUT_RULES_VERSION}|{SYSTEM}|{USER_TEMPLATE}"
                       .encode()).hexdigest()[:12]
    return f"x-{PROMPT_VERSION}{INPUT_RULES_VERSION}-{h}"

def extractor_version(model_id: str, revision: str) -> str:
    recovery = f"{RECOVERY_VERSION}|{MAX_NEW_TOKENS}|{RETRY_MAX_TOKENS}|{RETRY_HINT}"
    h = hashlib.sha256(f"{model_id}@{revision}|{PROMPT_VERSION}|{INPUT_RULES_VERSION}|{SYSTEM}|{USER_TEMPLATE}"
                       f"|{recovery}".encode()).hexdigest()[:12]
    return f"x-{PROMPT_VERSION}{INPUT_RULES_VERSION}{RECOVERY_VERSION}-{h}"

def compatible_versions(model_id: str, revision: str) -> list[str]:
    return [_r1_version(model_id, revision)]

def grounding_version() -> str:
    return f"{nz.version()}-{GROUNDING_RULES}"

LABEL = {"caption": "caption", "transcript": "spoken", "ocr": "on-screen", "ocr_title": "on-screen title"}

def prepare(record: dict, source: str = "instagram") -> dict:
    chunks = chunking.build_chunks(record, source)
    passages, context = [], []
    for c in chunks:
        md = c["metadata"]
        kind = md["content_type"]
        if kind == "ocr_fragment":
            continue
        if kind == "ocr_title" and md["n_words"] < MIN_EVIDENCE_WORDS:
            context.append(c["text"])
            continue
        passages.append({"pid": f"P{len(passages) + 1}", "chunk_id": c["id"], "content_type": kind,
                         "text": c["text"], "start": md.get("start"), "end": md.get("end")})
    return {"reel_id": record["reel_id"], "content_hash": chunking.content_hash(chunks),
            "passages": passages, "context": context}

def prompt_for(prepared: dict) -> str:
    def when(p):
        return f" {p['start']:.0f}-{p['end']:.0f}s" if p.get("start") is not None else ""
    lines = [f"{p['pid'][1:]}. ({LABEL[p['content_type']]}{when(p)}) {' '.join(p['text'].split())}"
             for p in prepared["passages"]]
    ctx = ("\nCONTEXT ONLY (short on-screen overlays; do not cite): " + " | ".join(prepared["context"][:20])
           if prepared["context"] else "")
    return USER_TEMPLATE.format(types=", ".join(nz.CONCEPT_TYPES),
                                predicates="\n".join(f"  {k}: {v}" for k, v in PREDICATES.items()),
                                max_c=MAX_CONCEPTS, max_r=MAX_RELATIONS, passages="\n".join(lines), context=ctx)

class ExtractionError(RuntimeError):

    def __init__(self, msg: str, attempts: list[dict] | None = None):
        super().__init__(msg)
        self.attempts = attempts or []

# Structural repair only: no name, quote, number or object is ever added, completed or guessed.
_REPAIRS = (
    ("inserted missing '{' before an object", re.compile(r'(\}\s*,\s*|\[\s*)("(?:n|s)"\s*:)'), r"\1{\2"),
    ('fixed misplaced quote \'"{n":\'', re.compile(r'"\{(n|s)"\s*:'), r'{"\1":'),
    ("removed stray quote after '}'", re.compile(r'\}"\s*,'), "},"),
    ("removed trailing comma", re.compile(r",\s*([\]}])"), r"\1"),
)

def repair_json(text: str) -> tuple[str, list[str]]:
    applied = []
    for label, rx, repl in _REPAIRS:
        text, n = rx.subn(repl, text)
        if n:
            applied.append(f"{label} ×{n}")
    return text, applied

def parse_with_repair(text: str) -> tuple[dict, list[str]]:
    try:
        return parse_json(text), []
    except ExtractionError as first:
        fixed, repairs = repair_json(text)
        if not repairs:
            raise
        try:
            return parse_json(fixed), repairs
        except ExtractionError as e:
            raise ExtractionError(f"{first} (repair failed: {e})") from e

def salvage(text: str) -> tuple[dict, dict]:
    """Truncated output: keep only the fully emitted objects, in order; nothing is completed or guessed."""
    t, repairs = repair_json(text)
    dec = json.JSONDecoder()
    rows: dict[str, list] = {"c": [], "r": []}
    info: dict = {"repairs": repairs}
    for key in ("c", "r"):
        m = re.search(rf'"{key}"\s*:\s*\[', t)
        if not m:
            info[f"{key}_list_missing"] = True
            continue
        i, complete = m.end(), False
        while i < len(t):
            while i < len(t) and t[i] in " \t\r\n,":
                i += 1
            if i < len(t) and t[i] == "]":
                complete = True
                break
            try:
                obj, i = dec.raw_decode(t, i)
            except json.JSONDecodeError:
                break
            rows[key].append(obj)
        info[f"{key}_list_complete"] = complete
    out = {"concepts": [x for x in map(_concept_from_row, rows["c"]) if x],
           "relations": [x for x in map(_relation_from_row, rows["r"]) if x]}
    info.update(salvaged_concepts=len(out["concepts"]), salvaged_relations=len(out["relations"]))
    return out, info

def parse_json(text: str) -> dict:
    """Requires both a "concepts" and a "relations" list; anything else is an error, never a silent empty extraction."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ExtractionError("no JSON object in model output")
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise ExtractionError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ExtractionError("JSON is not an object")
    if "c" in data or "r" in data:
        missing = [k for k in ("c", "r") if not isinstance(data.get(k), list)]
        if missing:
            raise ExtractionError(f"JSON lacks required list(s): {', '.join(missing)}")
        return {"concepts": [_concept_from_row(x) for x in data["c"]],
                "relations": [_relation_from_row(x) for x in data["r"]]}
    missing = [k for k in ("concepts", "relations") if not isinstance(data.get(k), list)]
    if missing:
        raise ExtractionError(f"JSON lacks required list(s): {', '.join(missing)}")
    return {"concepts": data["concepts"], "relations": data["relations"]}

def _pid(x) -> str | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return f"P{x}"
    if isinstance(x, str) and x.strip():
        t = x.strip().upper()
        return t if t.startswith("P") else f"P{t}"
    return None

def _concept_from_row(row) -> dict | None:
    if isinstance(row, dict):
        pids = row.get("p") if isinstance(row.get("p"), list) else []
        return {"name": row.get("n"), "type": row.get("t"), "surface": row.get("s"),
                "passages": [p for p in map(_pid, pids) if p]}
    if not isinstance(row, list) or len(row) < 3:
        return None
    pids = row[3] if len(row) > 3 and isinstance(row[3], list) else []
    return {"name": row[0], "type": row[1], "surface": row[2], "passages": [p for p in map(_pid, pids) if p]}

def _relation_from_row(row) -> dict | None:
    if isinstance(row, dict):
        return {"subject": row.get("s"), "predicate": row.get("p"), "object": row.get("o"),
                "passage": _pid(row.get("i")) or "", "quote": row.get("q")}
    if not isinstance(row, list) or len(row) < 5:
        return None
    return {"subject": row[0], "predicate": row[1], "object": row[2], "passage": _pid(row[3]) or "", "quote": row[4]}

_ATTEMPT_FIELDS = ("seconds", "prompt_tokens", "generation_tokens", "prompt_tps", "generation_tps",
                   "peak_memory_gb", "finish_reason")

def propose(llm, prepared: dict) -> tuple[dict, dict]:
    if not prepared["passages"]:
        return {"concepts": [], "relations": []}, {"seconds": 0, "skipped": "no evidence passages"}
    user = prompt_for(prepared)
    attempts: list[dict] = []

    def run(prompt: str, max_tokens: int, label: str) -> tuple[str, dict, dict]:
        text, st = llm.generate(SYSTEM, prompt, max_tokens=max_tokens)
        rec = {"attempt": label, "max_tokens": max_tokens, **{k: st.get(k) for k in _ATTEMPT_FIELDS if k in st}}
        attempts.append(rec)
        return text, st, rec

    def stats(last: dict, **extra) -> dict:
        out = {**last, "seconds": round(sum(a.get("seconds") or 0 for a in attempts), 2), "attempts": attempts,
               "recovery": RECOVERY_VERSION, **extra}
        out["peak_memory_gb"] = max((a.get("peak_memory_gb") or 0 for a in attempts), default=None)
        return out

    def finish(text: str, st: dict, rec: dict, **extra) -> tuple[dict, dict] | None:
        try:
            data, repairs = parse_with_repair(text)
        except ExtractionError as e:
            rec["outcome"] = f"invalid: {e}"
            return None
        rec["outcome"] = "repaired" if repairs else "ok"
        return data, stats(st, **({"repairs": repairs} if repairs else {}), **extra)

    text, st, rec = run(user, MAX_NEW_TOKENS, "first")
    if st.get("finish_reason") == "length":
        rec["outcome"], first = "truncated", f"output truncated at {MAX_NEW_TOKENS} tokens"
    else:
        done = finish(text, st, rec)
        if done:
            return done
        first = rec["outcome"]
    text, st, rec = run(user + RETRY_HINT, RETRY_MAX_TOKENS, "retry")
    if st.get("finish_reason") != "length":
        done = finish(text, st, rec, retried=first)
        if done:
            return done
        raise ExtractionError(f"{rec['outcome']} (after retry; first attempt: {first})", attempts)
    rec["outcome"] = "truncated"
    data, info = salvage(text)
    if data["concepts"]:
        return data, stats(st, retried=first, partial=True, salvage=info)
    raise ExtractionError(f"output truncated at {RETRY_MAX_TOKENS} tokens after retry and nothing complete to keep "
                          f"(first attempt: {first})", attempts)

def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold().replace("’", "'")
    s = re.sub(r"[^\w+#.\s]", " ", s)
    s = re.sub(r"(?<!\w)\.|\.(?!\w)", " ", s)
    return " ".join(s.split())

_EDGE = re.compile(r"^[^\w.#+]+|[^\w+#]+$")

def mention_tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s).casefold().replace("’", "'").replace("–", "-").replace("—", "-")
    out = []
    for raw in re.split(r"[\s/,;:!?()\[\]{}\"“”|]+", s):
        t = _EDGE.sub("", raw)
        if t.endswith("'s"):
            t = t[:-2]
        t = t.rstrip(".")
        if not t:
            continue
        parts = t.split("-")
        if len(parts) > 1 and all(p.isalpha() and len(p) >= 2 for p in parts):
            out += parts
        else:
            out.append(t)
    return out

def _fuzzy_allowed(surface: str, stoks: list[str]) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", surface)
    return (all(t.isalpha() for t in stoks) and sum(map(len, stoks)) >= FUZZY_MIN_CHARS
            and not (letters.isupper() and len(letters) <= 6))

def surface_match(surface: str, text: str) -> str | None:
    st, tt = mention_tokens(surface), mention_tokens(text)
    if not st:
        return None
    n = len(st)
    windows = [tt[i:i + n] for i in range(max(0, len(tt) - n + 1))]
    if any(w == st for w in windows):
        return "exact"
    if not _fuzzy_allowed(surface, st):
        return None
    s_key = nz.canonicalize(surface)[0]
    curated = nz._alias_table()["alias"]
    for w in windows:
        if any(a != b and (a.startswith(b) or b.startswith(a)) for a, b in zip(st, w)):
            continue
        if not all(x.isalpha() for x in w):
            continue
        if SequenceMatcher(None, " ".join(st), " ".join(w)).ratio() < FUZZY:
            continue
        w_key = nz.base_key(" ".join(w))
        if w_key in curated and nz.canonicalize(" ".join(w))[0] != s_key:
            continue
        return "fuzzy"
    return None

def assertion_id(reel_id: str, chunk_id: str, subj: str, pred: str, obj: str) -> str:
    return hashlib.sha1(f"{reel_id}|{chunk_id}|{subj}|{pred}|{obj}".encode()).hexdigest()[:20]

def _str(x) -> str:
    return x.strip() if isinstance(x, str) else ""

def recall_forms(entry: dict) -> list[str]:
    return sorted(set(entry["surfaces"]) | {entry["name"]} | set(nz.curated_surfaces(entry["name"])))

_SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n+|\s+[•·]\s+")
_LIST_WORDS = frozenset(("and", "or", "plus", "as", "well", "also", "nor"))
_CUES = frozenset("vs versus than unlike instead rather whereas but compared compare comparison differ different "
                  "difference alternative alternatives replace replaces replaced similar".split())
_NUMERIC = re.compile(r"^[$€£¥]?\s*\d[\d.,]*\s*[kKmMbB%]?\+?$")

def _items(sentence: str) -> list[str | None]:
    out: list[str | None] = []
    for part in re.split(r"([,;|])", sentence.replace(" & ", " and ")):
        if part in (",", ";", "|"):
            out.append(None)
        else:
            out += mention_tokens(part)
    return out

def _spans(items: list[str | None], key: str, entry: dict) -> list[tuple[int, int]]:
    out = set()
    for f in recall_forms(entry):
        ft = mention_tokens(f)
        n = len(ft)
        out |= {(i, i + n) for i in range(len(items) - n + 1) if n and items[i:i + n] == ft}
    for n in range(1, 6):
        for i in range(len(items) - n + 1):
            w = items[i:i + n]
            if None not in w and nz.canonicalize(" ".join(w))[0] == key:
                out.add((i, i + n))
    return sorted(out)

def _is_list(between: list[str | None]) -> bool:
    groups, cur = [], []
    for x in between:
        if x is None:
            groups.append(cur)
            cur = []
        else:
            cur.append(x)
    groups.append(cur)
    content = [[w for w in g if w not in _LIST_WORDS] for g in groups]
    if len(groups) == 1:
        return not content[0]
    return all(len(g) <= 3 for g in content)

def relation_evidence(quote: str, s_key: str, s_entry: dict, o_key: str, o_entry: dict) -> str | None:
    if not _spans(_items(quote), s_key, s_entry) or not _spans(_items(quote), o_key, o_entry):
        return "quote does not name both concepts exactly"
    listed = nested = False
    for sentence in _SENTENCE.split(quote):
        items = _items(sentence)
        ss, os_ = _spans(items, s_key, s_entry), _spans(items, o_key, o_entry)
        pairs = [(a, b) for a in ss for b in os_ if a[1] <= b[0] or b[1] <= a[0]]
        if not pairs:
            nested = nested or bool(ss and os_)
            continue
        if any(w in _CUES for w in items if w):
            return None
        if all(_is_list(items[min(a[1], b[1]):max(a[0], b[0])]) for a, b in pairs):
            listed = True
            continue
        return None
    if listed:
        return "quote only lists the concepts (enumeration, not a stated relation)"
    if nested:
        return "one concept is named only inside the other's name"
    return "quote names the concepts in different sentences"

def ground(raw: dict, prepared: dict) -> dict:
    rid = prepared["reel_id"]
    by_pid = {p["pid"]: p for p in prepared["passages"]}
    acronyms: dict[str, str] = {}
    for p in prepared["passages"]:
        acronyms.update(nz.find_acronyms(p["text"]))
    concepts: dict[str, dict] = {}
    mentions: dict[tuple[str, str], dict] = {}
    name_to_key: dict[str, str] = {}
    context_mapped: set[tuple[str, str]] = set()
    rejected: list[dict] = []

    for c in (raw.get("concepts") or [])[:MAX_CONCEPTS * 2]:
        if not isinstance(c, dict):
            continue
        name, surface = _str(c.get("name")), _str(c.get("surface")) or _str(c.get("name"))
        if not name:
            continue
        if nz.is_generic(name):
            rejected.append({"kind": "concept", "item": name, "reason": "generic"})
            continue
        if nz.excluded(name) or nz.excluded(surface):
            rejected.append({"kind": "concept", "item": name, "reason": f"curated non-concept: "
                             f"{nz.excluded(name) or nz.excluded(surface)}"})
            continue
        cited = [by_pid[p] for p in (c.get("passages") or []) if isinstance(p, str) and p in by_pid]

        def where(term):
            hits = [(p, m) for p in cited if (m := surface_match(term, p["text"]))]
            return hits or [(p, m) for p in prepared["passages"] if (m := surface_match(term, p["text"]))]

        k_surface, d_surface = nz.canonicalize(surface, model_canonical=name, acronyms=acronyms)
        k_name, d_name = nz.canonicalize(name, acronyms=acronyms)
        candidates = []
        if k_name == k_surface:
            found = where(surface) or where(name)
            if found:
                candidates.append((k_name, d_name, surface, found))
        else:
            by_name = where(name)
            if by_name:
                candidates.append((k_name, d_name, name, by_name))
            by_surface = where(surface)
            if by_name and surface_match(name, surface):
                by_surface = []
            if by_surface and len(surface.split()) > MAX_SURFACE_WORDS:
                rejected.append({"kind": "concept", "item": surface, "reason": "surface is a phrase, not a concept name"})
            elif by_surface and not any(p["content_type"] in SPOKEN_OR_CAPTION for p, _ in by_surface):
                rejected.append({"kind": "concept", "item": surface, "reason": "uncorroborated OCR surface (model "
                                 f"read it as {name!r}, which the text does not contain)"})
            elif by_surface and not nz.is_generic(surface):
                candidates.append((k_surface, d_surface, surface, by_surface))
                rejected.append({"kind": "rename", "item": f"{surface} → {name}",
                                 "reason": f"model name not supported for this surface; kept {d_surface!r} as its own concept"})
        if not candidates:
            rejected.append({"kind": "concept", "item": name, "surface": surface,
                             "reason": "surface not found in any evidence passage"})
            continue
        split = []
        for key, display, used, found in candidates:
            spec = nz.context_alias(used)
            if spec is None:
                split.append((key, display, used, found))
                continue
            hit = [(p, m) for p, m in found if nz.context_supports(spec, p["text"])]
            miss = [(p, m) for p, m in found if not nz.context_supports(spec, p["text"])]
            if hit:
                split.append((nz.base_key(spec["canonical"]), spec["canonical"], used, hit))
                context_mapped.update((p["chunk_id"], nz.base_key(spec["canonical"])) for p, _ in hit)
            if miss:
                split.append((key, display, used, miss))
        candidates = split
        for key, display, used, found in candidates:
            if nz.is_generic(display):
                rejected.append({"kind": "concept", "item": display, "reason": "generic"})
                continue
            if _NUMERIC.match(display) or not any(w.isalpha() and w.lower() not in nz.CONNECTORS
                                                  for w in re.findall(r"[^\W\d_]+", display)):
                rejected.append({"kind": "concept", "item": display, "reason": "numeric value, not a concept name"})
                continue
            if (nz.curated_type(display) is None and len(re.sub(r"[\W_]", "", used)) <= 2
                    and all(p["content_type"] not in SPOKEN_OR_CAPTION for p, _ in found)):
                rejected.append({"kind": "concept", "item": display,
                                 "reason": "short OCR token (≤ 2 characters, not curated, only on screen)"})
                continue
            ctype = nz.curated_type(display) or (c.get("type") if c.get("type") in nz.CONCEPT_TYPES else "concept")
            entry = concepts.setdefault(key, {"key": key, "name": display, "type": ctype, "surfaces": []})
            if used not in entry["surfaces"]:
                entry["surfaces"].append(used)
            name_to_key[nz.base_key(used)] = key
            if key == k_name:
                name_to_key[nz.base_key(name)] = key
            for f in nz.curated_surfaces(display):
                name_to_key.setdefault(nz.base_key(f), key)
            for p, m in found:
                ctx = "+context_alias" if (p["chunk_id"], key) in context_mapped else ""
                mentions.setdefault((p["chunk_id"], key), {"chunk_id": p["chunk_id"], "key": key, "surface": used,
                                                           "method": f"llm+{m}_match{ctx}"})

    for key, entry in concepts.items():
        forms = recall_forms(entry)
        for p in prepared["passages"]:
            if (p["chunk_id"], key) in mentions:
                continue
            for f in forms:
                spec = nz.context_alias(f)
                if spec and (nz.base_key(spec["canonical"]) == key) != nz.context_supports(spec, p["text"]):
                    continue
                if surface_match(f, p["text"]) == "exact":
                    mentions[(p["chunk_id"], key)] = {"chunk_id": p["chunk_id"], "key": key, "surface": f,
                                                      "method": "string_match"}
                    break

    def resolve(n: str) -> str | None:
        k = name_to_key.get(nz.base_key(n))
        if k:
            return k
        k = nz.canonicalize(n, acronyms=acronyms)[0]
        return k if k in concepts else None

    assertions: dict[str, dict] = {}
    for r in (raw.get("relations") or [])[:MAX_RELATIONS * 2]:
        if not isinstance(r, dict):
            continue
        subj_n, obj_n = _str(r.get("subject")), _str(r.get("object"))
        pred = _str(r.get("predicate")).upper().replace(" ", "_")
        quote, pid = _str(r.get("quote")), _str(r.get("passage"))
        item = f"{subj_n} {pred} {obj_n}"

        def reject(reason):
            rejected.append({"kind": "relation", "item": item, "quote": quote, "reason": reason})

        if pred not in PREDICATES:
            reject("predicate not in vocabulary")
            continue
        s_key, o_key = resolve(subj_n), resolve(obj_n)
        if s_key is None or o_key is None:
            reject("subject or object is not a grounded concept")
            continue
        if s_key == o_key:
            reject("self-relation")
            continue
        p = by_pid.get(pid)
        if p is None:
            reject("cited passage is not an evidence passage")
            continue
        nq = norm_text(quote)
        if not (MIN_QUOTE_WORDS <= len(nq.split()) <= MAX_QUOTE_WORDS):
            reject("quote length out of range")
            continue
        if f" {nq} " not in f" {norm_text(p['text'])} ":
            reject("quote not found verbatim in the cited passage")
            continue
        if (p["chunk_id"], s_key) not in mentions or (p["chunk_id"], o_key) not in mentions:
            reject("subject/object not both mentioned in the cited passage")
            continue
        why = relation_evidence(quote, s_key, concepts[s_key], o_key, concepts[o_key])
        if why:
            reject(why)
            continue
        aid = assertion_id(rid, p["chunk_id"], s_key, pred, o_key)
        assertions.setdefault(aid, {"id": aid, "reel_id": rid, "chunk_id": p["chunk_id"], "subject": s_key,
                                    "predicate": pred, "object": o_key, "quote": quote,
                                    "method": "llm_proposed+quote_verified",
                                    "confidence": "quote_names_both",
                                    "provenance": "quote_verified", "status": "proposed"})

    by_statement: dict[tuple, list[dict]] = {}
    for a in assertions.values():
        by_statement.setdefault((frozenset((a["subject"], a["object"])), a["chunk_id"], norm_text(a["quote"])),
                                []).append(a)
    for group in by_statement.values():
        preds = sorted({a["predicate"] for a in group})
        if len(preds) > 1:
            for a in group:
                del assertions[a["id"]]
                rejected.append({"kind": "relation", "item": f"{a['subject']} {a['predicate']} {a['object']}",
                                 "quote": a["quote"], "reason": "several predicates proposed for one statement "
                                 f"({', '.join(preds)}): relation type undetermined"})
    return {"concepts": concepts, "mentions": sorted(mentions.values(), key=lambda m: (m["chunk_id"], m["key"])),
            "assertions": sorted(assertions.values(), key=lambda a: a["id"]), "rejected": rejected,
            "normalizer_version": grounding_version()}