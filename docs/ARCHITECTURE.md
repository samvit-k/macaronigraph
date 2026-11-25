# MacaroniGraph architecture

Technical reference for the capture pipeline, storage, search index, knowledge graph and answer generation.
The [README](../README.md) gives the overview, installation and usage; [EVALUATION.md](EVALUATION.md) the measured results.

## Job tracking (SQLite `output/pipeline.db`, the source of truth)

| Reel state | Meaning |
|---|---|
| `pending` | not attempted yet (or released without a counted attempt) |
| `in_progress` | leased by a running capture session |
| `queued` | uploaded, waiting for or in background processing |
| `done` | best record is `ok`, `ok_no_speech` or an acceptable `partial` (e.g. truncated at 180 s) |
| `needs_retry` | a counted attempt failed in a retryable way |
| `exhausted` | 3 counted attempts used; the best attempt is kept |
| `failed_permanent` | "page not available" confirmed twice |
| `skipped` | not a Reel (`/p/` image posts) |

- **Resume:**
  - a session leases the Reels it queues;
  - un-uploaded leases are released when the session finishes;
  - a crashed session's leases are reclaimed after 10 minutes without a heartbeat (the extension sends one every 30 s).
  - Releasing restores each Reel's previous state, so a `done` Reel leased by a "reprocess all" run stays `done`.
- **Retries:**
  - Stop, login walls and lost capture don't use up a retry;
  - a good result is never replaced by a worse re-run.
- **Idempotent:** a repeated upload for the same session and Reel is recognised and not processed twice.
- **Outputs:**
  - `output/enriched_reels.jsonl` has **one canonical record per Reel** (its best attempt) plus `processing.history`. It's refreshed every 10 attempts or 60 s, at session end, and on demand.
  - `output/attempts.jsonl` (and the `attempts` table) holds every attempt.

## Background processing (`backend/worker.py`)

The extension no longer waits for Whisper and OCR. Each upload is saved to `output/spool/` and processed by one background worker, so the next Reel is captured while the previous one is processed.

- **Atomic hand-off:** uploads are written to `.incoming-*` first. The Reel is marked `queued` and the job is published (renamed) under one lock, so reconciliation never sees a half-published job.
- **Nothing lost on database errors:**
  - the extracted record is checkpointed (`record.json`) before the SQLite write;
  - media is deleted only after the attempt is safely stored;
  - a database failure keeps the job and retries it without redoing extraction.
- **Quarantine:**
  - After 3 failed writes, unreadable metadata, or a worker error, the job moves to `spool/.dead/` and SQLite is reconciled immediately.
  - Media is deleted at quarantine **only** if a valid, well-typed checkpoint holds the extracted record. Otherwise the media is the only recoverable copy and is kept until you run `python -m backend.worker purge-dead --older-than DAYS`.
  - `retry-dead` re-queues quarantined jobs. A corrupt checkpoint is set aside and never reused.
- **Crash recovery at startup:**
  - half-written uploads are removed;
  - jobs already committed before a crash are cleaned up without a duplicate attempt;
  - other jobs resume;
  - `queued` Reels without a job return to their previous state.
- **Sessions:** a session is finalized only when none of its uploads are still queued, and a session that is still draining is never treated as abandoned.
- **Backpressure:** if more than 2 jobs are waiting, uploads wait. If the backlog is still full after 15 minutes, the upload is refused with 503 (nothing is saved) and the extension stops the session; un-uploaded Reels go back to the queue.
- `ASYNC_PROCESSING=0` processes each upload inline using the same code. The tests use this mode, apart from dedicated async tests.

## Output record

| field | content |
|---|---|
| `reel_id`, `source_url`, `saved_at`, `source_index`, `session_id`, `processed_at` | identity and provenance |
| `status` | `ok` · `ok_no_speech` (silent or music only) · `partial` · `failed`; anything other than `ok` comes with `errors[]`; `warnings[]` lists non-fatal notes (e.g. `briefly_muted`) |
| `metadata` | caption (mojibake fixed), title, hashtags, owner, brand partner, and `raw`, an exact copy of the original export entry |
| `capture` | full duration and capture limit, seconds captured, how sound was enabled, muted flag **and muted seconds**, frames attempted/sampled/dropped, actual interval, crop, whether the stream was still live, user-activation state |
| `transcript` | `audio_class` (speech/silent/non_speech/missing), language, `text` (confident segments only), `uncertain_text`, segments with timestamps, confidence and flags, rejected segments with reasons |
| `ocr` | `segments` (cleaned), `raw_segments` (every tracked line with bounding box), `filtered_segments` (noise, kept), frame stats |
| `passages` | `chunk_id = <reel>:<caption|transcript|ocr>:<n>`, source type, timestamps, text, URL; OCR passages also carry `ocr_kind`, `ocr_parts`, `ocr_echo`, `ocr_quality`. This is the search-index input |
| `processing` | (canonical file only) state, counted attempts, canonical attempt id, full attempt history |

### OCR clean-up (no deletions: everything raw stays in the record)

1. **Noise filter:** junk lines ("RQ", "a a a", "©") go to `filtered_segments`. Confident short terms ("AI", "x86", "WSL") are kept.
2. **Blocks:** lines of one caption or paragraph in the same frame are grouped together.
3. **Speech echo:** each segment is labelled `speech_echo`, `partial_echo` or `on_screen` by comparing it with the words Whisper heard around the same time. This is a label, not proof, and nothing is removed because of it.
4. **Caption merging:** short-lived fragments merge into one `subtitle` segment only when they match the speech **and** share band, font size and horizontal centre. The merged segment carries `parts` (raw ids) and `aligned_speech`. Unrelated on-screen text is never merged.
5. **Repeats:** recurring text (titles, list markers) becomes one segment with every `occurrence`.
6. **Fragment labels:** isolated one-frame scraps of at most 2 words, with no protected token and a non-word or low confidence, get `quality: "fragment"`. They stay searchable and are ranked lower. Protected tokens are digits, file names and paths, capitalised names, confident acronyms, real words, recurring terms, spoken words, and words from the caption or hashtags.

## Semantic search

SQLite stays the source of truth. The Chroma index in `output/chroma/` is a derived cache: deleting it is always safe, and the next sync rebuilds it from SQLite.

**What gets indexed.** Every Reel whose canonical record is `ok`, `ok_no_speech` or `partial` is chunked by `backend/chunking.py` (`CHUNKER_VERSION`):
| Chunk type | What it contains |
|---|---|
| `caption` | The caption. |
| `transcript` | Whisper segments merged into windows of about 50–120 words, with one segment of overlap. |
| `ocr_title` | Overlays visible for a long time, stored once. |
| `ocr` | A "screen": on-screen lines visible around the same time, in reading order, so lists and code stay intact. |
| `ocr_fragment` | OCR labelled uncertain. Still searchable, but its score is multiplied by `FRAGMENT_WEIGHT` (0.85). |

OCR that repeats speech is dropped only when *confident* transcript segments at the same moment cover its words. Persistent overlays and Reels without reliable speech keep all their OCR. Each chunk lists its source `passage_ids` and its start/end times.

**Model.** `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, 1024-dimensional vectors) runs locally through sentence-transformers on MPS, with the Hugging Face revision pinned. It is downloaded once (~1.2 GB) and runs offline after that. Change it with `EMBED_MODEL`.

**Rebuilds.** The index records a *signature*: model, revision, dtype, sequence length and chunker version. When the signature changes:
1. A new collection is built while the old one keeps serving search.
2. Before switching, the new collection is caught up with SQLite and fully verified.
3. If verification fails, the old collection stays active.
4. The replaced collection is dropped after a grace period.

**Incremental and independent of capture.** The backend's indexer thread syncs automatically, but only when nothing else is happening:
- no capture session is open;
- the spool is empty;
- there have been 120 s of quiet (`INDEX_IDLE_S`).

A sync re-embeds only Reels whose canonical record changed and skips everything else. If capture resumes mid-sync, the sync stops before its next embedding batch, and finished Reels stay committed. Manual sync is always allowed. Set `INDEX_AUTO=0` to turn off automatic syncing.

```bash
curl 'localhost:8765/api/search?q=how+do+I+learn+GPU+kernels&k=5'   # grouped per Reel; &group=none for raw passages
curl 'localhost:8765/api/search?q=ROS2+navigation&types=ocr'         # types: caption,transcript,ocr,ocr_title,ocr_screen,ocr_fragment
curl localhost:8765/api/index/status
curl -X POST 'localhost:8765/api/index/sync'                         # ?rebuild=1 forces a fresh, verified collection
.venv/bin/python -m backend.indexer status|verify|search "query"     # sync/rebuild only while the backend is stopped
.venv/bin/python scripts/search_eval.py                              # retrieval eval (needs a private query set: fixtures/search_queries.json)
```

- The index build peaks at about 4 GB of memory, and the first search after a restart takes about 5 s while the model loads. Retrieval results are in [EVALUATION.md](EVALUATION.md).

## Knowledge graph

The three stores have distinct jobs:
- **SQLite** holds the canonical records and is the source of truth.
- **ChromaDB** answers "which passages are relevant?"
- **Neo4j** answers "how are these things connected, and what supports that?"

Neo4j is optional: capture, processing and semantic search work unchanged when it is stopped.

**Setup.** No sudo; everything is installed inside the project.
```bash
scripts/neo4j.sh install   # Neo4j Community (GPLv3) → .tools/neo4j, sha256-verified; password → .env.local (mode 600)
scripts/neo4j.sh start     # 127.0.0.1 only, bolt 7697 / http 7484; data in output/neo4j (derived, rebuildable)
.venv/bin/python scripts/check_env.py --graph
```
Credentials are read only from `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD`. `.env.local` fills them when they are unset, and nothing is hardcoded.

**Schema.**
```
(:Reel {reel_id})-[:HAS_PASSAGE]->(:Passage {chunk_id = Chroma chunk id})-[:MENTIONS {surface, method}]->(:Concept {key})
(:Assertion {id, predicate, quote, status, method, confidence})-[:SUBJECT|OBJECT]->(:Concept), -[:EVIDENCE]->(:Passage)
```
- Passages use exactly the Chroma chunk IDs, so a vector hit maps straight into the graph.
- Assertions are nodes, so each piece of supporting evidence keeps its own quote, passage, method and status.
- Co-occurrence ("in the same Reel") is computed at query time and is never stored as a relationship.
- Embedding similarity only produces *suggestions* (`candidates`) and is never written to the graph.

**Extraction: a local LLM proposes, deterministic code verifies.**
- The model is `mlx-community/Qwen3-4B-Instruct-2507-4bit` (Apache-2.0, 2.3 GB) running through mlx-lm on this Mac. No content leaves the machine.
- **Concepts** are kept only if their surface form actually occurs in an evidence passage:
  - identifier-aware matching keeps CUDA ≠ CUDA-Q, C ≠ C++ ≠ C#, and x86 ≠ x86-64;
  - fuzzy matching applies only to plain words, for OCR and ASR errors;
  - fragments, and short overlays such as "SHIP IT", are never evidence;
  - OCR-only spellings the model disagrees with are rejected.
- **Relations** are kept only if:
  - the predicate is in a closed vocabulary (`IS_A`, `PART_OF`, `USES`, `DEPENDS_ON`, `USED_FOR`, `ALTERNATIVE_TO`, `CONTRASTS_WITH`, `PREREQUISITE_FOR`);
  - the quote is verbatim in the cited passage;
  - the quote itself names both concepts.
- Every rejection is kept with its reason.
- **Normalization:** case, spacing, ordinary plurals, curated synonyms in `backend/graph/aliases.json`, acronyms defined in the same Reel, and guarded ASR corrections (for example "Cloud Code" → Claude Code, only when the model agrees).

**A quote proves provenance, not truth.**

| Relation status | Meaning |
|---|---|
| `proposed` | Model-extracted, quote verified. |
| `corroborated` | The same claim was proposed independently in two or more Reels. |
| `verified` / `rejected` | Your decision, via `python -m backend.graph review <assertion_id> accept|reject [note]`. |

Reviews are stored in SQLite and re-applied on every rewrite, so they survive refreshes and rebuilds.

**Sync.**
- **Change detection:** keyed on canonical attempt, content hash, extractor version and normalizer version. Unchanged Reels are skipped with no LLM call.
- **Extraction cache:** results are cached in SQLite, so Neo4j can be rebuilt without the LLM, and editing `aliases.json` re-normalizes without re-extracting.
- **Writes:** each Reel is written in one Neo4j transaction that replaces only what that Reel owns. Concepts shared with other Reels survive.
- **Failures:** a failed extraction or write leaves the previous graph untouched.
- **Chroma coordination:** a Reel is `ready` for GraphRAG only while every chunk is in the *active* Chroma collection with the same content hash; otherwise it is `awaiting_index`. Readiness is rechecked on every sync, in both directions.
- **Concurrency:** one sync runs at a time, across processes too. Automatic sync runs only while capture and processing are idle and never overlaps Chroma embedding. Manual sync always writes cached extractions, but runs new LLM extraction during capture only with `--force`.

**Commands.**
```bash
python -m backend.graph status | sync [--force] | verify
python -m backend.graph reel <reel_id> | concept <name> | shared [min] | between <a> <b> | evidence <a> <b>
python -m backend.graph review <assertion_id> accept|reject [note] | candidates [threshold]
curl localhost:8765/api/graph/status        # also /verify, /reel/{id}, /concept?name=, /shared, /between?a=&b=, /evidence?a=&b=
curl -X POST 'localhost:8765/api/graph/sync'            # ?force=1, ?wait=1; returns a request id → /api/graph/request/{id}
```

## GraphRAG

Open `http://localhost:8765/app/` (served by the backend; no build step). Everything runs locally: Qwen3-Embedding-0.6B for the query, Neo4j for graph expansion, Qwen3-4B (MLX, 4-bit) for answers. No external API is called.

- **Retrieval** (`backend/rag.py`, modes `vector | graph | hybrid`, default hybrid): Chroma passages, plus passages of other Reels that share a specific concept (corpus-relative specificity ≥ 0.75) or carry evidence of a proposed relation. Every graph-found passage must pass a relevance gate and be retrieval-ready (present in the active Chroma collection with a matching content hash). Channels are fused by weighted RRF; the top vector Reels keep their ranks. If the graph is down or any graph step fails, retrieval falls back to vector and says so (`fallback`).
- **Answers** (`backend/answer.py`): at most 6 sentences, each with citations to the numbered sources; sentences without a valid citation, or with a quoted span (double or single quotes) that is not verbatim in its cited source, are removed. Lexical support is shown as a diagnostic only. If nothing survives, the answer is the fixed "not enough information" message.
- **Conversations** (`backend/conversations.py`, separate `output/conversations.db`): follow-ups are rewritten from previous *questions* and concepts of *cited passages* only — never from assistant text. Answers never enter SQLite records, Chroma or Neo4j.
- **Scheduling** (`backend/ask.py`, `backend/models.py`): one shared model behind a cross-process heavy-work lock; during capture or while the model is busy, generation is queued (max 8) and the page polls; queued items expire visibly on restart.
- Parameters were tuned on the dev questions only and frozen in `fixtures/rag_params.json` (= `rag.DEFAULTS`).

```bash
curl -X POST localhost:8765/api/ask -H 'content-type: application/json' \
     -d '{"question":"Which Reels mention NeetCode?","mode":"hybrid"}'   # also conversation_id, generate, options
curl localhost:8765/api/ask/{request_id}            # poll a queued answer
curl localhost:8765/api/conversations               # list; GET/DELETE /api/conversations/{id}
.venv/bin/python scripts/rag_eval.py retrieval|tune|answers|baseline --split dev|heldout   # needs private question sets
```


## Answer generation

Retrieval parameters are frozen in `fixtures/rag_params.json`. The answer configuration is frozen in `fixtures/rag_answer_config.json` (configuration **D6**); `answer.DEFAULTS` and `conversations.DEFAULTS` are tested to equal it. Every component below is an independent switch, so earlier behaviour can be restored for comparison.
- **Citations:** also accepted inside the sentence text (`[S2]`, `(S1, S3)`, `c:["S1"]`). Each marker counts only for its own sentence, and malformed ids (`[2]`, `Source 2`, `S0`, ids outside the pack) are rejected and reported. The citation, inference and verbatim-quote checks are unchanged (no quote repair).
- **Prompt v2:** answer directly, cover every part, name each Reel for list questions, paraphrase and quote only short exact phrases, keep negations and numbers exact, state missing parts instead of abstaining, and use no outside knowledge.
- **OCR fragments:** passages with fewer than 3 real words are not shown to the model.
- **Absence statements** ("X is not mentioned in the sources") are not presented as answers; they become the fixed "part not covered" note, or "not enough information". A Reel's own negative claim ("you do not need X") is kept as an answer.
- **Evidence gate:** abstains without generating only if the best vector similarity is < 0.55 **and** a question term occurs in no retrieved passage. It is never applied to follow-up turns.
- **Follow-ups (rewrite v2):**
  - one entity named in earlier questions plus "it/this/that" → substitution;
  - otherwise the question is used with earlier questions as context, and the model is never called for rewriting;
  - topic changes and follow-ups that name their own subject are left alone.
- **Answer model:** `RAG_LLM` (environment variable, default = `GRAPH_LLM` = Qwen3-4B-Instruct-2507-4bit). Only one model is resident at a time. `RAG_LLM=mlx-community/Qwen3-8B-4bit` is supported (pinned revision, thinking disabled) but **not recommended** on 24 GB: on dev it was not more accurate, was 1.3–2.5× slower, and swapped.

- **Deterministic listings and counts** (`backend/listing.py`): "Which Reels mention X?" and "How many Reels mention X?" are answered by exact search, without the model, also while capture is running. Matches are word-boundary literal mentions (with spacing, hyphen and camel-case variants; graph aliases only when one concept covers the whole term) in the active index, each validated against the current canonical record and content hash. Answers distinguish transcript and caption from on-screen text, cite one passage per Reel, report the searched corpus size, and say that only literal mentions are listed.
- **Passage selection** (`backend/evidence.py`): adds up to four answer-bearing passages from Reels already in the evidence pack (lexical and embedding similarity; no new Reels, nothing reordered or removed).
- **Clarification** (`backend/conversations.py`): a singular "it/this/that" after a question that named two or more things returns a clarifying question instead of a guess, unless the follow-up uses comparison wording.
- **Claim verification** (`backend/verify.py`): implemented but **off** in D6. Modes `flag | drop | repair` (at most four model calls per answer; a repaired claim is never delivered without re-verification). Measured as not worth its cost with the 4B model; see [EVALUATION.md](EVALUATION.md).

## Workspace API (read-only additions)

| Endpoint | Purpose |
|---|---|
| `GET /api/graph/overview` | Paged Reels, concepts mentioned by two or more Reels, their mention edges and non-rejected relationships, with totals |
| `GET /api/graph/neighborhood?kind=reel\|concept&id=` | One node's neighbourhood with passage excerpts, for progressive expansion and the inspector |
| `GET /api/graph/search?q=` | Concepts by name or alias, and Reels by creator or caption |
| `DELETE /api/conversations` | Clears all conversations (conversation history only; Reel data is untouched) |
| `GET /welcome`, `GET /welcome/extension.zip`, `/workspace/` | Welcome page, the extension packaged from an explicit file allowlist, and the built workspace |

Graph views live in `backend/graph/view.py` and never write. The workspace never loads the whole graph: it starts from the overview and expands one neighbourhood at a time.

```bash
.venv/bin/python scripts/data_digest.py --with-verify --out digest.json    # read-only integrity digest of SQLite, Chroma and Neo4j
.venv/bin/python scripts/data_digest.py --compare before.json after.json  # IDENTICAL, or the exact differences
.venv/bin/python scripts/answer_eval.py run --set v2dev --config D6 --out run.json   # needs the private question sets
```

