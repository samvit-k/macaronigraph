# MacaroniGraph

**A knowledge graph of your saved Instagram Reels.**

MacaroniGraph turns the Reels you save into searchable passages (transcripts, captions and on-screen text), links
them through the concepts they mention, and answers your questions with citations to the original Reels.
Transcription, text recognition, embeddings, the graph and the AI answers all run locally on a Mac.

- **Capture:** a Chrome extension plays your saved Reels in a worker tab and records audio and sampled frames.
- **Extraction:** MLX-Whisper transcribes speech and Tesseract reads on-screen text. Both produce one validated
  record per Reel, stored in SQLite.
- **Search:** an incremental Chroma index of transcript, caption and on-screen-text passages, embedded with
  Qwen3-Embedding-0.6B.
- **Knowledge graph:** Neo4j holds Reels, passages and concepts, plus relationships extracted by a local model that
  must quote the passage they come from.
- **Answers:** hybrid vector-and-graph retrieval feeds a local Qwen3-4B model. Every sentence cites its sources, and
  listing and count questions are answered by exact search instead of the model.
- **Workspace:** a React and Cytoscape app with an interactive graph, an evidence inspector and a chat that
  highlights the cited Reels.

## Architecture

```
Chrome extension (MV3)                    Local backend: FastAPI on 127.0.0.1:8765
 popup → background.js ─ tabCapture ─►    /api/process → spool → background worker
         content.js (worker tab)            ffmpeg → MLX-Whisper → Tesseract → record → SQLite (source of truth)
         offscreen.js (recorder, frames)    indexer  → chunking → Qwen3-Embedding → Chroma      (derived)
                                            graph    → Qwen3-4B extraction → grounding → Neo4j  (derived, optional)
Browser: /workspace/  (React + Cytoscape)   /api/ask → hybrid retrieval → D6 answer pipeline → cited answer
         /welcome     (extension download)  /api/graph/* read-only views · conversations in a separate SQLite file
```

- **Sources of truth:** SQLite holds every Reel and attempt. Chroma and Neo4j are rebuilt from it, and deleting them is
  always safe.
- **Heavy work:** a cross-process lock lets one heavy job run at a time (transcription, embedding, extraction or
  answering). While capture or processing is running, answers are queued and filled in when the model is free.
- **Graceful degradation:** if Neo4j is stopped, retrieval falls back to vector search and says so.
- **More detail:** the full technical reference is [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Privacy and data

- **What runs locally:** all AI inference and knowledge processing — transcription, OCR, embeddings, graph
  extraction and answering. No cloud AI API is called, and the backend listens on `127.0.0.1` only.
- **What uses the network:**
  - capture loads instagram.com in your own logged-in browser;
  - `scripts/setup.sh` downloads ffmpeg, Tesseract and Python packages;
  - `scripts/neo4j.sh install` downloads Neo4j;
  - the first run downloads the Whisper, embedding and language models from Hugging Face;
  - `npm ci` downloads the frontend packages.
- **Where your data lives:** everything MacaroniGraph produces stays in `output/` (records, index, graph,
  conversations), which is git-ignored. So are your Instagram data export and credentials (`.env.local`).
- **Terms of Service:** Instagram's Terms of Use prohibit automated access. MacaroniGraph only visits your own saved
  items, slowly and in your own session, but it is still automated access. Each processed Reel counts as a play.
  Nothing is liked, saved, followed or posted.

## Requirements

- macOS on Apple Silicon (MLX), with about 10 GB free for models and tools.
- Google Chrome 116 or later.
- [uv](https://docs.astral.sh/uv/), `curl`, and Node.js 20+ with npm (for the workspace).
- Optional, for the knowledge graph: Java 21 or 25. Neo4j Community itself is installed inside the project by `scripts/neo4j.sh`.

## Installation

```bash
git clone https://github.com/samvit-k/macaronigraph.git && cd macaronigraph
./scripts/setup.sh                                    # ffmpeg + Tesseract → .tools/env, Python 3.11 deps → .venv (no sudo)
.venv/bin/python scripts/check_env.py --whisper       # verifies ffmpeg, Tesseract and MLX-Whisper (downloads the model once)
scripts/neo4j.sh install && scripts/neo4j.sh start    # optional knowledge graph (127.0.0.1 only; password → .env.local)
cd frontend && npm ci && npm run build && cd ..       # builds the workspace into backend/static/workspace
```

## Usage

1. **Start the backend:** `.venv/bin/uvicorn backend.app:app --port 8765`.
2. **Install the extension:**
   1. Open `http://127.0.0.1:8765/welcome` and click **Download Extension (.zip)**, or use the `extension/` folder directly.
   2. In `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and choose the folder.
3. **Point it at your saved Reels:**
   1. Request your Instagram data export.
   2. Place the unzipped `instagram-*` folder in the project folder or in `~/Downloads`, or set `SAVED_POSTS_PATH` to
      its `saved_posts.json`. Only `/reel/` links are queued.
4. **Capture:**
   1. Open instagram.com while logged in, as the visible tab of its window.
   2. In the extension popup choose the source and limits, then click **Start Processing**.
   3. Click the **▶ Start Processing** button on the page.

   Each Reel is captured in real time; processing continues in the background.
5. **Explore:** open `http://127.0.0.1:8765/workspace/`.
   - **Overview:** Reels and the concepts several of them share.
   - **Expand:** double-click a node to show everything connected to it.
   - **Inspect:** click a node or edge for source passages and the original Instagram link.
   - **Ask:** use the chat to ask questions. Citations open the exact passage and highlight it in the graph.

Useful commands:

```bash
.venv/bin/python -m backend.jobs status             # Reel states, attempts, open sessions
.venv/bin/python -m backend.indexer status|verify   # search index
.venv/bin/python -m backend.graph status|sync|verify
curl -X POST 127.0.0.1:8765/api/ask -H 'content-type: application/json' -d '{"question":"Which Reels mention Python?"}'
```

### Configuration

Settings come from environment variables; nothing is hardcoded. The main ones:

| Variable | Default | Purpose |
|---|---|---|
| `OUTPUT_DIR` | `output/` | Where records, index, graph data and conversations are stored |
| `SAVED_POSTS_PATH` | auto-detected | Path to the Instagram export's `saved_posts.json` |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | `bolt://127.0.0.1:7697`, `neo4j`, from `.env.local` | Knowledge graph connection |
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model (pinned revision) |
| `RAG_LLM` / `GRAPH_LLM` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | Answer and extraction model |
| `ASYNC_PROCESSING`, `INDEX_AUTO`, `GRAPH_AUTO` | `1` | Background processing, automatic indexing, automatic graph sync |

The answer pipeline's settings are frozen in `fixtures/rag_answer_config.json` (configuration **D6**), and retrieval
parameters in `fixtures/rag_params.json`. Tests check that the code defaults match both files.

## Testing

```bash
.venv/bin/python -m pytest tests -q          # 671 tests; live Neo4j tests use a throwaway instance (skipped if Neo4j is not installed)
node --test tests/ext/*.test.js              # extension tests
cd frontend && npm test                      # 46 Vitest tests: graph model, layout, viewport, chat state, inspector
cd frontend && node e2e/workspace.e2e.mjs    # real-browser checks against a running backend with processed Reels
```

- **Isolation:** every Python test runs against a temporary `OUTPUT_DIR`. A session guard refuses to open the
  production database, Chroma directory or Neo4j instance.
- **First run:** some tests use the real Whisper and embedding models, which are downloaded on first use.
- **Browser checks:** these use the installed Google Chrome through `playwright-core`, with no browser download. They
  are data-agnostic: their topics come from whatever graph the backend serves, and they delete only the
  conversations they create.

## Evaluation

Measured on the author's saved Reels. The datasets are private, so these results cannot be reproduced from this
repository alone.

| | Result |
|---|---|
| Hybrid retrieval, held-out | Recall@5 0.977, evidence-pack coverage 0.984 (vector only: 0.833 and 0.850) |
| Answers, sealed held-out set (53 turns) | **75.5% fully correct** (previous configuration: 60.4%); citation validity 100% |
| Faithfulness | 7.8% of answer sentences unsupported |
| Latency | median warm answer 5.2 s on an Apple Silicon Mac |

Methodology, every metric, the failures and what can be reproduced publicly are in [docs/EVALUATION.md](docs/EVALUATION.md).
All answer judgments are self-assessed by the developer, not independent.

## Known limitations

- **Answers:**
  - **Correctness:** about 1 in 4 answers is incomplete or wrong, and 7.8% of answer sentences are unsupported.
  - **Abstention:** questions the Reels can't answer, but that are close to their topics, are sometimes answered from
    adjacent or garbled text instead of being declined.
  - **Citations:** a citation shows where a statement comes from; it does not guarantee the statement is correct.
- **Listing and count questions:** phrasings outside the recognised patterns fall back to the model, which can miscount.
  Exact search lists literal mentions only, and transcription misspellings are not aliased.
- **Capture:**
  - Capture runs in real time and needs the worker tab visible and the Mac awake.
  - Reels longer than 180 s are truncated and marked `partial`.
  - Image posts are skipped.
  - Instagram may briefly re-mute playback, which is detected and recorded.
- **Recognition:** OCR misreads stylised or animated text, and Whisper can mishear brand names. On-screen text is
  labelled in answers and the inspector.
- **Relationships:** the graph is mostly Reel–concept mentions. Relationships between concepts are sparse, and stay
  "proposed" until they are corroborated by several Reels or reviewed by you
  (`python -m backend.graph review <id> accept|reject`).
- **Platform:** the extension is not on the Chrome Web Store and works only with this local backend, which needs macOS
  on Apple Silicon.

## Project structure

```
backend/      FastAPI app, capture processing (jobs, worker, transcribe, ocr, record), search (chunking, embedder, indexer),
              answers (rag, ask, answer, listing, evidence, verify, conversations), graph/ (extract, normalize, sync, queries, view)
extension/    MV3 Chrome extension: background, content and offscreen scripts, popup (ui-next/)
frontend/     MacaroniGraph workspace and welcome page (React, Cytoscape, d3-force, Vite) and browser checks (e2e/)
scripts/      setup, Neo4j installer, environment check, test-media generator, validation, evaluation and integrity tools
fixtures/     synthetic test media and ground truth, frozen retrieval and answer configurations
tests/        pytest suite (+ tests/ext for the extension)
docs/         architecture reference and evaluation summary
```

## License

No license has been granted yet. All rights reserved.
