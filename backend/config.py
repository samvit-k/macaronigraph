"""Central configuration from environment variables; all paths are absolute and derived from the project root."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS_BIN = ROOT / ".tools" / "env" / "bin"

def _tool(name: str) -> str:
    local = TOOLS_BIN / name
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which(name) or str(local)

# Libraries that shell out to bare `ffmpeg` (e.g. mlx_whisper.audio) must find the local build too.
if TOOLS_BIN.is_dir() and str(TOOLS_BIN) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{TOOLS_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

FFMPEG = os.environ.get("FFMPEG_BIN", _tool("ffmpeg"))
TESSERACT = os.environ.get("TESSERACT_BIN", _tool("tesseract"))
TESSDATA = os.environ.get("TESSDATA_PREFIX", str(ROOT / ".tools" / "env" / "share" / "tessdata"))

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "output"))
OUTPUT_JSONL = OUTPUT_DIR / "enriched_reels.jsonl"
TMP_ROOT = Path(os.environ.get("REEL_TMP_ROOT", ROOT / ".tmp"))

TEST_MEDIA_DIR = ROOT / "fixtures" / "media"
TEST_POSTS_PATH = ROOT / "fixtures" / "test_saved_posts.json"
TEST_GROUND_TRUTH = ROOT / "fixtures" / "ground_truth.json"
def _find_export() -> Path:
    if os.environ.get("SAVED_POSTS_PATH"):
        return Path(os.environ["SAVED_POSTS_PATH"])
    rel = "instagram-*/your_instagram_activity/saved/saved_posts.json"
    found = sorted([*ROOT.glob(rel), *(Path.home() / "Downloads").glob(rel)], key=lambda p: p.stat().st_mtime)
    return found[-1] if found else ROOT / "saved_posts.json"

INSTAGRAM_POSTS_PATH = _find_export()

PORT = int(os.environ.get("PORT", "8765"))
ASYNC_PROCESSING = os.environ.get("ASYNC_PROCESSING", "1") == "1"
PIPELINE_VERSION = "phase2-0.1"

EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "mps")
EMBED_DTYPE = os.environ.get("EMBED_DTYPE", "float32")
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "16"))
INDEX_AUTO = os.environ.get("INDEX_AUTO", "1") == "1"
INDEX_IDLE_S = float(os.environ.get("INDEX_IDLE_S", "120"))
FRAGMENT_WEIGHT = float(os.environ.get("FRAGMENT_WEIGHT", "0.85"))

def chroma_dir() -> Path:
    return OUTPUT_DIR / "chroma"

def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.startswith("NEO4J_") and k not in os.environ:
            os.environ[k] = v

_load_env_file(ROOT / ".env.local")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7697")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
GRAPH_LLM = os.environ.get("GRAPH_LLM", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
GRAPH_SOURCES = tuple(s for s in os.environ.get("GRAPH_SOURCES", "instagram").split(",") if s)
GRAPH_AUTO = os.environ.get("GRAPH_AUTO", "1") == "1"

RAG_LLM = os.environ.get("RAG_LLM", GRAPH_LLM)
MODEL_REVISIONS = {"mlx-community/Qwen3-8B-4bit": "545dc4251c05440727734bcd94334791f6ab0192"}
RAG_IDLE_UNLOAD_S = float(os.environ.get("RAG_IDLE_UNLOAD_S", "300"))
RAG_HEAVY_WAIT_S = float(os.environ.get("RAG_HEAVY_WAIT_S", "90"))
LISTING_SUMMARY = os.environ.get("LISTING_SUMMARY", "1") == "1"

def configure_tesseract() -> None:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = TESSERACT
    if Path(TESSDATA).is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", TESSDATA)