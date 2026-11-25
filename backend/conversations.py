"""Phase 4B conversation memory — a SEPARATE SQLite file (OUTPUT_DIR/conversations.db), never read by the graph
sync or the semantic index. Assistant answers are stored here only: they never become graph assertions, index
entries or extraction cache rows.

Follow-up rewriting uses only (a) the user's earlier questions' concepts and (b) concepts mentioned in the
passages the previous answer CITED (validated graph mentions) — never the assistant's prose:
  1. deterministic: if the previous question named exactly one concept, a pronoun ("this", "it", "that") is
     replaced by its name;
  2. otherwise the local model rewrites the question, and the rewrite is accepted only if every content word comes
     from the question or from the allowed referent names;
  3. otherwise (model busy/deferred/invalid) the question is kept as written and the referents become retrieval
     hints (boost_concepts).
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid

from . import config, db, models

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL, question TEXT NOT NULL, standalone_query TEXT, rewrite TEXT, mode_requested TEXT,
  mode_used TEXT, status TEXT NOT NULL, error TEXT, retrieval TEXT, answer TEXT, question_concepts TEXT,
  cited_concepts TEXT, cited_reels TEXT, timing TEXT, created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS turns_conv ON turns(conversation_id, seq);
"""
_lock = threading.Lock()
_FOLLOWUP = re.compile(r"\b(it|its|this|that|these|those|they|them|their|previous|other|both|the two|either|"
                       r"the same|which of them|which of these|which of those)\b", re.I)
_PRONOUN = re.compile(r"\b(this|it|that)\b(?!\s+(?:reel|video)s?\b)", re.I)
_OTHER_REELS = re.compile(r"\bother (?:saved )?(?:reels?|videos?)\b", re.I)
_SCOPE_PREVIOUS = re.compile(r"\bwhich of (?:them|these|those)\b", re.I)
_FUNCTION = frozenset("""a an the and or but of in on at by for from with about as to into over is are was were be
do does did what which who how why when where whose that this these those it its they them their other others reels
reel videos video my saved mention mentions mentioned talk talks say says compare compared comparison between
difference differ than with versus vs explain relationship related relate same both either one ones any more also
prerequisites prerequisite learning learn learned can could would should i me you your there""".split())

def db_path():
    return config.OUTPUT_DIR / "conversations.db"

def connect():
    return db.connect(db_path(), SCHEMA, foreign_keys=True)

def tx():
    return db.transaction(_lock, connect)

_JSON = ("retrieval", "answer", "question_concepts", "cited_concepts", "cited_reels", "timing", "rewrite")

def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for k in _JSON:
        if k in d and d[k] is not None:
            d[k] = json.loads(d[k])
    return d

def create(title: str | None = None) -> str:
    cid = uuid.uuid4().hex[:12]
    with tx() as con:
        con.execute("INSERT INTO conversations(id, title, created_at, updated_at) VALUES(?,?,?,?)",
                    (cid, (title or "")[:80], db.utc_now(), db.utc_now()))
    return cid

def exists(cid: str) -> bool:
    with connect() as con:
        return con.execute("SELECT 1 FROM conversations WHERE id=?", (cid,)).fetchone() is not None

def list_conversations(limit: int = 50) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute(
            """SELECT c.*, (SELECT COUNT(*) FROM turns t WHERE t.conversation_id=c.id) AS turns
               FROM conversations c ORDER BY updated_at DESC LIMIT ?""", (limit,))]

def get(cid: str) -> dict | None:
    with connect() as con:
        c = con.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
        if c is None:
            return None
        turns = [_row(r) for r in con.execute("SELECT * FROM turns WHERE conversation_id=? ORDER BY seq", (cid,))]
    return {**dict(c), "turns": turns}

def delete(cid: str) -> bool:
    with tx() as con:
        con.execute("DELETE FROM turns WHERE conversation_id=?", (cid,))
        return con.execute("DELETE FROM conversations WHERE id=?", (cid,)).rowcount == 1

def delete_all() -> int:
    """Delete every conversation and its turns (the workspace's "Clear all"). Only conversations.db is touched;
    a queued answer for a deleted turn is dropped by the answer queue (its turn no longer exists)."""
    with tx() as con:
        con.execute("DELETE FROM turns")
        return con.execute("DELETE FROM conversations").rowcount

def add_turn(cid: str, question: str, **fields) -> int:
    with tx() as con:
        seq = con.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM turns WHERE conversation_id=?", (cid,)).fetchone()[0]
        cols = ["conversation_id", "seq", "question", "created_at", "updated_at", *fields]
        vals = [cid, seq, question, db.utc_now(), db.utc_now(),
                *[json.dumps(v) if k in _JSON and v is not None else v for k, v in fields.items()]]
        tid = con.execute(f"INSERT INTO turns({', '.join(cols)}) VALUES({', '.join('?' * len(cols))})", vals).lastrowid
        con.execute("UPDATE conversations SET updated_at=? WHERE id=?", (db.utc_now(), cid))
    return tid

def update_turn(tid: int, **fields) -> None:
    sets = ", ".join(f"{k}=?" for k in fields) + ", updated_at=?"
    vals = [json.dumps(v) if k in _JSON and v is not None else v for k, v in fields.items()]
    with tx() as con:
        con.execute(f"UPDATE turns SET {sets} WHERE id=?", (*vals, db.utc_now(), tid))

def get_turn(tid: int) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM turns WHERE id=?", (tid,)).fetchone()
    return _row(r) if r else None

def previous_turns(cid: str, n: int = 2) -> list[dict]:
    """The last n turns, most recent first."""
    with connect() as con:
        return [_row(r) for r in con.execute(
            "SELECT * FROM turns WHERE conversation_id=? ORDER BY seq DESC LIMIT ?", (cid, n))]

def expire_queued(reason: str = "the backend restarted before this answer was generated") -> int:
    """Startup: queued generations of a previous process are marked expired (visible), never silently lost."""
    with tx() as con:
        return con.execute("UPDATE turns SET status='expired', error=?, updated_at=? WHERE status='queued'",
                           (reason, db.utc_now())).rowcount

# ---------------------------------------------------------------- follow-ups

def is_followup(question: str) -> bool:
    return bool(_FOLLOWUP.search(question))

# Phase 4C rewrite configuration (selected on dev data).
BASE_DEFAULTS = {"referents": "v1", "clarify": False}   # clarify: Phase 4D disambiguation (v2 only)
DEFAULTS = {**BASE_DEFAULTS, "referents": "v2", "clarify": True}   # Phase 4C/4D selection (fixtures/rag_answer_config.json)

def referents(turns: list[dict]) -> list[dict]:
    """Allowed referents, most relevant first: concepts of the previous questions, then concepts mentioned in the
    passages the previous answers cited. Never derived from assistant text."""
    out: list[dict] = []
    for t in turns:
        for src, items in (("question", t.get("question_concepts") or []), ("cited", t.get("cited_concepts") or [])):
            for c in items:
                if c["key"] not in {x["key"] for x in out}:
                    out.append({"key": c["key"], "name": c["name"], "from": src, "turn": t["seq"]})
    return out[:8]

def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9+#.\-']*[a-z0-9+#]|[a-z0-9]", s.lower())

def _allowed(question: str, refs: list[dict]) -> set[str]:
    return set(_words(question)) | {w for r in refs for w in _words(r["name"])} | _FUNCTION

REWRITE_SYSTEM = ("You rewrite a follow-up question into a standalone question. You never answer it and never "
                  "add information.")
REWRITE_USER = """Previous question: {prev}
Follow-up question: {q}
Names the follow-up may refer to (use only these to replace words like "it", "that", "these", "the other"):
{names}
Return ONE line of JSON: {{"q": "<the follow-up rewritten as a standalone question>"}}
Keep the user's wording and intent. Do not add any other names, facts or assumptions."""

def rewrite(question: str, turns: list[dict], llm=None, allow_model: bool = True,
            timeout: float | None = None) -> dict:
    """→ {standalone, method: none|substitution|model|fallback, referents, boost, exclude_previous,
          scope_previous, reason}"""
    t0 = time.perf_counter()
    prev = turns[0] if turns else None
    refs = referents(turns)
    out = {"standalone": question, "method": "none", "referents": refs, "boost": [],
           "exclude_previous": bool(_OTHER_REELS.search(question)) and prev is not None,
           "scope_previous": bool(_SCOPE_PREVIOUS.search(question)) and prev is not None,
           "previous_cited_reels": (prev or {}).get("cited_reels") or [], "reason": None}
    if DEFAULTS.get("referents") == "v2":
        if prev:
            out.update(_rewrite_v2(question, turns, out))
        out["rewrite_s"] = round(time.perf_counter() - t0, 4)
        return out
    if not prev or not is_followup(question) or not refs:
        out["rewrite_s"] = round(time.perf_counter() - t0, 4)
        return out
    q_refs = [r for r in refs if r["from"] == "question" and r["turn"] == prev["seq"]]
    if len(q_refs) == 1 and _PRONOUN.search(question):
        out.update(standalone=_PRONOUN.sub(q_refs[0]["name"], question, count=1), method="substitution")
    elif allow_model:
        try:
            out.update(_model_rewrite(question, prev, refs, llm, timeout))
        except Exception as e:  # noqa: BLE001 — busy model, unreadable output: fall back, never fail
            out.update(method="fallback", reason=f"{type(e).__name__}: {e}")
    else:
        out.update(method="fallback", reason="model unavailable (capture/processing active)")
    if out["method"] == "fallback":
        out["boost"] = [r["key"] for r in refs[:3]]
    out["rewrite_s"] = round(time.perf_counter() - t0, 4)
    return out

# ---------------------------------------------------------------- Phase 4C rewriting (v2)
# Recent user-named entities are a strong preference, not an absolute rule; concepts that merely appear in cited
# passages are never substituted; ambiguity is never forced into a substitution — the user's own words are kept and
# the earlier QUESTIONS (never answers) are added as context for retrieval and for the answer model.

_REF = re.compile(r"\b(it|its|this|that|these|those|they|them|their|the two|either|both)\b"
                  r"(?!\s+(?:reel|video|creator|step|part|one)s?\b)", re.I)
_OWN_SUBJECT = re.compile(r"\bthe\s+(?:[\w'’\-]+\s+){1,6}(?:reel|video)\b", re.I)
_TOPIC_CHANGE = re.compile(r"\b(switching topics?|change of topic|different (?:topic|question)|unrelated|new topic)\b",
                           re.I)

def _question_referents(turns: list[dict]) -> list[dict]:
    out: list[dict] = []
    for t in turns:                                              # most recent first
        for c in t.get("question_concepts") or []:
            if c["key"] not in {x["key"] for x in out}:
                out.append({"key": c["key"], "name": c["name"], "turn": t["seq"]})
    return out

_SINGULAR = re.compile(r"\b(it|its|this|that)\b(?!\s+(?:reel|video|creator|step|part|one)s?\b)", re.I)
# Comparison/contrast wording means the user is talking about BOTH things named before → no clarification needed.
_CONTRAST = re.compile(r"\b(?:compare[sd]?|comparison|other|both|versus|vs\.?|differ(?:s|ence|ent)?|between|each)\b", re.I)
_STEP_REF = re.compile(r"\b(?:that|this|the same)\s+(?:step|part|stage|round|one)\b", re.I)

def _rewrite_v2(question: str, turns: list[dict], out: dict) -> dict:
    q = _SCOPE_PREVIOUS.sub(" ", question) if out["scope_previous"] else question   # "which of them" → scoping
    if _TOPIC_CHANGE.search(question):
        return {"method": "none", "reason": "topic change"}
    if _OWN_SUBJECT.search(question):
        return {"method": "none", "reason": "the follow-up names its own subject"}
    clarify = DEFAULTS.get("clarify")
    if clarify and _STEP_REF.search(q):
        # "that step" / "this part": the user's earlier questions say which step — keep them as context, never guess
        earlier = [t["question"] for t in turns if t.get("question")]
        return {"method": "context", "standalone": f"{question} (Earlier questions in this conversation: "
                + "; ".join(f'"{x}"' for x in earlier) + ")", "retrieval_query": " ".join([question] + earlier),
                "boost": [c["key"] for c in _question_referents(turns)[:3]], "reason": "refers to an earlier step/part"}
    if not _REF.search(q):
        return {"method": "none", "reason": "nothing to resolve"}
    cands = _question_referents(turns)
    if clarify and _SINGULAR.search(q) and not _CONTRAST.search(q) and not _REF.sub("", q).strip(" ?") == "":
        last = [c for c in cands if c["turn"] == turns[0]["seq"]]
        if len(last) >= 2:
            # one singular pronoun, several things named in the immediately previous question, none named now:
            # ask instead of guessing (Phase 4D). The question is kept as written.
            return {"method": "clarify", "standalone": question, "clarify": [c["name"] for c in last[:4]],
                    "reason": "several possible referents in the previous question"}
    if len(cands) == 1 and _PRONOUN.search(q):
        return {"method": "substitution", "standalone": _PRONOUN.sub(cands[0]["name"], question, count=1),
                "reason": "single entity named in earlier questions"}
    earlier = [t["question"] for t in turns if t.get("question")]
    context = "; ".join(f'"{x}"' for x in earlier)
    return {"method": "context", "standalone": f"{question} (Earlier questions in this conversation: {context})",
            "retrieval_query": " ".join([question] + earlier), "boost": [c["key"] for c in cands[:3]],
            "reason": ("several possible referents" if len(cands) > 1 else
                       "no entity named in earlier questions" if not cands else "reference is not a simple pronoun")}

def _model_rewrite(question, prev, refs, llm, timeout) -> dict:
    llm = llm or models.shared_llm()
    user = REWRITE_USER.format(prev=prev["question"], q=question, names="\n".join(f"- {r['name']}" for r in refs))
    with models.interactive():
        with models.heavy("follow-up rewriting", timeout=config.RAG_HEAVY_WAIT_S if timeout is None else timeout):
            text, _ = llm.generate(REWRITE_SYSTEM, user, max_tokens=80)
    m = re.search(r"\{.*\}", text or "", re.S)
    new = json.loads(m.group(0)).get("q") if m else None
    if not isinstance(new, str) or not new.strip():
        return {"method": "fallback", "reason": "rewrite unreadable"}
    new = " ".join(new.split())
    extra = [w for w in _words(new) if w not in _allowed(question, refs)]
    if extra:
        return {"method": "fallback", "reason": f"rewrite added words not in the question or referents: {extra[:5]}"}
    return {"standalone": new, "method": "model"}