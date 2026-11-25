# Evaluation

This page summarises how MacaroniGraph was measured during development, from capture through answer quality. It
reports aggregate numbers only.

## What can and cannot be reproduced from this repository

| Result | Data it needs | Reproducible from the public repository? |
|---|---|---|
| Unit, integration and browser tests | Synthetic fixtures (`fixtures/media`, `ground_truth.json`, `test_saved_posts.json`) | **Yes** |
| Test-media capture validation (WER, OCR card recall) | Generated test clips (`scripts/make_test_media.py`) | **Yes** |
| OCR clean-up, reliability batch, search, retrieval and answer results below | The author's saved Reels and evaluation sets built from them | **No.** The Reels, Reel IDs, question sets, sealed held-out sets and judgments are private and are not published. |

- **Why the datasets are private:** they are built from one person's saved Instagram Reels, and contain Reel IDs, transcript quotes and facts from those Reels.
- **Running the evaluations yourself:** the scripts are included (`scripts/search_eval.py`, `scripts/rag_eval.py`, `scripts/answer_eval.py`) and run against your own collection once you create question sets in the same format.
- **How judgments were made:** all manual judgments of answers were made by the developer, not by an independent reviewer, and are labelled self-assessed.

## Capture and extraction

**Test media** (Chrome, 1000 ms frame interval; reproducible): every check passed.
- **Speech clips:** word error rate ≤ 0.03, and OCR found every on-screen card shown for at least 1 s.
- **Silent and music-only clips:** classified correctly.
- **Burned-in subtitles:** 10 fragments merged into 1 subtitle segment, with WER 0.04.
- **Short overlays:** a 0.6 s overlay is missed at 1000 ms, so use a 250–500 ms interval if short overlays matter.

**OCR clean-up v1 → v2** (5 real Reels; proxy metrics, because real Reels have no ground truth):

| | v1 | v2 |
|---|---|---|
| Informative tokens kept | 99% | 100% |
| Garbled segments | 17% | 11% |
| Duplicate segments | 12% | 0% |

**Reliability batch** (25 real Reels, 55 min):
- **Results:** 23 `ok`, 1 `ok_no_speech` and 1 `partial`.
- **Integrity:** no duplicates, no leftover leases or media, and no backend errors.
- **Speed:** about 2.2 min per Reel with inline processing, which motivated background processing.

## Semantic search

59 Reels, 1,255 chunks, 34 queries:

| Model | Hit@1 | Hit@5 | MRR | Embedding speed |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B (used) | 0.97 | 1.00 | 0.99 | 8 chunks/s |
| bge-base-en-v1.5 | 0.97 | 1.00 | 0.98 | 61 chunks/s |

The query set is too easy to separate the two models.

## GraphRAG retrieval

Held-out questions, parameters frozen on the dev split:

| Mode | Recall@5 | Hit@5 | MRR | Evidence-pack coverage |
|---|---|---|---|---|
| Vector only | 0.833 | 0.889 | 0.847 | 0.850 |
| Hybrid (vector + graph) | **0.977** | **1.000** | **0.875** | **0.984** |

These figures were reproduced identically after every later change to answer generation, so answer work never altered
retrieval.

## Answer quality

Answers come from a local Qwen3-4B-Instruct-2507 (4-bit) model, with temperature 0 and at most one model call per answer.

### Final configuration (D6) on a sealed held-out set

The set was written and sealed before any answers were generated on it, then run once per configuration. It has 53
turns: 42 answerable (including 2 that should be clarified) and 11 unanswerable.

| Metric | Previous configuration | **D6 (shipped)** |
|---|---|---|
| Fully correct (all turns) | 32/53 (60.4%) | **40/53 (75.5%)** |
| Fully correct (answerable turns) | 25/42 (59.5%) | **33/42 (78.6%)** |
| False abstentions | 3/42 (7.1%) | **1/42 (2.4%)** |
| Correct abstentions | 7/11 (63.6%) | 7/11 (63.6%) |
| Unsupported sentences | 6/46 (13.0%) | **6/77 (7.8%)** |
| Answers containing an unsupported claim | 6/43 (14.0%) | 6/43 (14.0%) |
| Listing and count completeness | 1/8 | **6/8** (every exact-search answer was complete) |
| Follow-up turns correct | 3/6 | **5/6** |
| Clarifications: appropriate / unnecessary | 0/2 / 0 | **2/2 / 0** |
| Citation validity | 100% | 100% |
| Warm answer latency, median / p90 | 5.4 s / 7.1 s | 5.2 s / 7.4 s |
| Peak memory (no pressure warnings) | 6.5 GB | 4.1 GB |

D6 was better than or equal to the previous configuration on every turn, and the gain seen during development held on
this sealed set.

**Remaining failures in D6:**
- **Near-topic traps:** 4 unanswerable questions were answered from adjacent or garbled text instead of abstaining.
- **Unrecognised listings:** 2 listing questions were phrased outside the patterns the system recognises.
- **Retrieval misses:** 3 answers missed a passage that was never retrieved.
- **Misreadings:** 3 answers misread a passage.
- **Incomplete:** 1 answer left out part of what was asked.

### Components that were measured and not shipped

- **Claim verification** by the same 4B model: precision 0.45 and recall 0.50 on labelled sentences, with about 3 s
  added per answer. It removed some correct content along with unsupported claims. It is implemented but turned off.
- **Larger answer model** (Qwen3-8B, 4-bit): no more accurate on development questions, 1.3–2.5× slower, and it
  caused swapping on a 24 GB machine.

### Targets not met

| Target | Status |
|---|---|
| Fewer than 5% unsupported sentences | **Not met (7.8%)** |
| At least 90% correct abstentions | **Not met (63.6%)** |
| 100% listing completeness | **Not met (75%)**: missed only because two phrasings weren't recognised |

## Data integrity during evaluation

Every evaluation ran read-only against production data. A digest of the SQLite records, the Chroma collection (every
table and every HNSW segment file) and the Neo4j graph was taken before and after each phase, and the digests were
identical each time. A test-session guard (`tests/conftest.py`) refuses to open the production database, Chroma
directory or Neo4j instance.
