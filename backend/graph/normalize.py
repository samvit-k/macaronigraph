"""Concept normalization: canonical keys, curated synonyms, acronym definitions, generic-word filtering.

Merge only clear duplicates — case, spacing, ordinary plurals, curated synonyms (aliases.json), acronyms
defined in the same Reel. Never fuzzy-merge different names:
  C ≠ C++ ≠ C#,  CUDA ≠ CUDA-Q,  x86 ≠ x86-64,  GPU ≠ TPU,  Pandas ≠ panda.

The normalization version (rules + aliases.json digest) is part of the graph hash, not of the LLM extraction
version: editing aliases.json re-normalizes and rewrites affected Reels without re-running the LLM.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata
from pathlib import Path

ALIASES_PATH = Path(__file__).with_name("aliases.json")
RULES_VERSION = "n1"

# Generic words / filler that are never concepts on their own (a name made only of these is rejected).
STOPLIST = frozenset("""
a an the of and or for to with in on at by from my your our their his her its all most some any every
thing things stuff video videos reel reels post posts tip tips trick tricks hack hacks way ways idea ideas
people person someone everyone guy guys friend friends you me i we they it this that these those
project projects work job jobs time day days year years today life world step steps part
problem problems question questions answer answers example examples result results lot lots kind type
something everything nothing anything one two three first second third best top new free good great
comment link links bio follow followers like likes share save content creator channel page account
resource resources tool tools technology technologies concept concepts method methods skill skills
career careers student students school college class company companies team teams money
""".split())
CONNECTORS = frozenset(("of", "and", "the", "for", "to", "in", "on", "a"))
CONCEPT_TYPES = ("technology", "tool", "method", "concept", "organization", "person", "resource")
PROTECTED_WORDS = frozenset("series species news physics mathematics analytics robotics electronics graphics "
                            "kubernetes pandas windows jenkins redis ios macos aws sass less express".split())
_LEAD = re.compile(r"^[^\w.#+]+")          # keep a leading "." (.NET)
_TRAIL = re.compile(r"[^\w+#]+$")           # keep trailing "+" / "#" (C++, C#)
_TOKEN_JUNK = re.compile(r"[^\w+#.\-]")

def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("’", "'").replace("–", "-").replace("—", "-")
    return " ".join(s.split())

def _singular(word: str) -> str:
    if len(word) <= 3 or not word.isalpha() or word in PROTECTED_WORDS:
        return word
    if word.endswith(("ss", "us", "is", "ous", "ics", "sis")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and word[:-2].endswith(("ch", "sh", "x")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word

def _tokens(name: str) -> list[tuple[str, str]]:
    """→ [(original token, normalized token)], hyphenated plain words split into words."""
    s = _fold(name).replace("&", " and ")
    s = _TRAIL.sub("", _LEAD.sub("", s))
    out: list[tuple[str, str]] = []
    for raw in re.split(r"[\s_/]+", s):
        tok = _TRAIL.sub("", _LEAD.sub("", raw))
        if tok.lower().endswith("'s"):
            tok = tok[:-2]
        tok = _TOKEN_JUNK.sub("", tok).rstrip(".")
        if not tok:
            continue
        parts = tok.split("-")
        if len(parts) > 1 and all(p.isalpha() and len(p) >= 2 for p in parts):
            out += [(p, p.lower()) for p in parts]      # "retrieval-augmented" → "retrieval augmented"
        else:
            out.append((tok, tok.lower()))              # identifiers keep hyphens: "cuda-q", "x86-64"
    return out

def base_key(name: str) -> str:
    toks = _tokens(name)
    if not toks:
        return ""
    words = [n for _, n in toks]
    orig = toks[-1][0]
    acronym_plural = len(orig) > 2 and orig[-1] == "s" and orig[:-1].isupper()
    title_phrase = len(toks) > 1 and all(o[:1].isupper() for o, _ in toks)
    if orig.islower() or acronym_plural or (len(toks) > 1 and not title_phrase):
        # never proper names: single capitalized words (Pandas, Windows) or Title-Case phrases (Linus Torvalds)
        words[-1] = _singular(words[-1]) if not acronym_plural else words[-1][:-1]
    return " ".join(words)

# ---------------------------------------------------------------- curated aliases (reloaded when the file changes)

_cache: dict = {"stamp": None}
_cache_lock = threading.Lock()

def _alias_table() -> dict:
    st = ALIASES_PATH.stat()
    stamp = (str(ALIASES_PATH), st.st_mtime_ns, st.st_size)
    with _cache_lock:
        if _cache["stamp"] != stamp:
            raw = ALIASES_PATH.read_bytes()
            doc = json.loads(raw)
            data = doc["concepts"]
            excluded = {base_key(n): why for n, why in doc.get("not_concepts", {}).items()}
            contextual = {base_key(s): {"surface": s, **spec} for s, spec in doc.get("contextual_aliases", {}).items()}
            alias, types, asr, literal = {}, {}, {}, {}
            for canon, spec in data.items():
                types[canon] = spec["type"]
                literal[canon] = [canon, *spec.get("aliases", [])]
                for a in [canon, *spec.get("aliases", [])]:
                    alias[base_key(a)] = canon
                for a in spec.get("asr_aliases", []):
                    asr[base_key(a)] = canon
            _cache.update(stamp=stamp, alias=alias, types=types, asr=asr, literal=literal, excluded=excluded,
                          contextual=contextual,
                          digest=hashlib.sha256(raw).hexdigest()[:12])
        return _cache

def version() -> str:
    """Normalization version: rules + current aliases.json content."""
    return f"{RULES_VERSION}-{_alias_table()['digest']}"

def is_generic(name: str) -> bool:
    k = base_key(name)
    if k in _alias_table()["alias"]:
        return False                         # curated concepts (incl. one-letter "C") are never generic
    if not k or len(k) < 2 or k.replace(".", "").isdigit():
        return True
    words = k.split()
    return all(w in STOPLIST for w in words) or len(words) > 6

def excluded(name: str) -> str | None:
    """Curated non-concepts (aliases.json "not_concepts", exact key match only) → the recorded reason."""
    return _alias_table()["excluded"].get(base_key(name))

def context_alias(surface: str) -> dict | None:
    """Curated context-dependent alias for this surface (aliases.json "contextual_aliases"), e.g. "Higgs Field"
    → Higgsfield only when its passage also names the company's products. None for every other surface."""
    return _alias_table()["contextual"].get(base_key(surface))

def context_supports(spec: dict, text: str) -> bool:
    """True if the passage contains one of the spec's required context words (whole tokens, case-insensitive).
    Deliberately narrow lists: a generic word such as "AI" alone is never one of them."""
    tokens = set(re.findall(r"[a-z0-9][a-z0-9.\-]*[a-z0-9]|[a-z0-9]", unicodedata.normalize("NFKC", text).casefold()))
    return any(w.casefold() in tokens for w in spec["requires_any"])

def contextual_pairs() -> set[frozenset]:
    """Key pairs that are deliberately distinct concepts despite similar spelling (surface vs its target)."""
    return {frozenset((k, base_key(s["canonical"]))) for k, s in _alias_table()["contextual"].items()}

def find_acronyms(text: str) -> dict[str, str]:
    """Acronyms defined in the text as '<words> (<ACRONYM>)', where the words end right before the
    parenthesis and their initials spell the acronym — either including connector words
    ('Simultaneous Localization and Mapping (SLAM)') or skipping them ('proximal policy optimization (PPO)',
    'return on investment (ROI)' / 'bill of materials (BOM)' both ways)."""
    out = {}
    for m in re.finditer(r"([A-Za-z][A-Za-z\- ]{2,80}?)\s*\(\s*([A-Z][A-Za-z]{1,7})\s*\)", text):
        words, acr = m.group(1).replace("-", " ").split(), m.group(2).lower()
        for k in range(2, min(len(acr) + 3, len(words)) + 1):
            cand = words[-k:]
            if cand[0].lower() in CONNECTORS or cand[-1].lower() in CONNECTORS:
                continue
            with_conn = "".join(w[0] for w in cand).lower()
            without = "".join(w[0] for w in cand if w.lower() not in CONNECTORS).lower()
            if acr in (with_conn, without):
                out[base_key(m.group(2))] = " ".join(cand)
                break
    return out

def canonicalize(name: str, model_canonical: str | None = None, acronyms: dict | None = None) -> tuple[str, str]:
    """→ (concept key, display name). Order: curated alias → guarded ASR alias → acronym defined in this Reel
    → base key. ASR aliases apply only when the model's canonical name agrees (e.g. 'Cloud Code' heard,
    model says 'Claude Code')."""
    t = _alias_table()
    k = base_key(name)
    if k in t["alias"]:
        canon = t["alias"][k]
        return base_key(canon), canon
    if k in t["asr"] and model_canonical and base_key(model_canonical) == base_key(t["asr"][k]):
        canon = t["asr"][k]
        return base_key(canon), canon
    if acronyms and k in acronyms:
        expanded = acronyms[k]
        ek = base_key(expanded)
        if ek in t["alias"]:
            return base_key(t["alias"][ek]), t["alias"][ek]
        return ek, expanded
    display = _fold(name)
    if "_" in display and " " not in display:
        display = display.replace("_", " ")            # "Pi_harness" → "Pi harness" (same key either way)
    return k, display

def curated_type(display: str) -> str | None:
    return _alias_table()["types"].get(display)

def curated_surfaces(display: str) -> list[str]:
    """Literal spellings (canonical name + aliases, as written in aliases.json) of a curated concept, for exact
    text matching. ASR aliases are excluded: they apply only with the model's agreement."""
    return list(_alias_table()["literal"].get(display, []))