"""Verify that ffmpeg, Tesseract and the Python stack actually work.

Usage: .venv/bin/python scripts/check_env.py [--whisper] [--embed] [--graph]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config  # noqa: E402

failures: list[str] = []

def check(name, fn):
    try:
        print(f"  ok   {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        failures.append(name)
        print(f"  FAIL {name}: {e}")

def ver(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return (out.stdout or out.stderr).splitlines()[0]

def ocr_roundtrip():
    from PIL import Image, ImageDraw, ImageFont
    import pytesseract

    config.configure_tesseract()
    img = Image.new("RGB", (640, 120), "white")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((20, 30), "HELLO TESSERACT", fill="black", font=font)
    txt = pytesseract.image_to_string(img).strip()
    assert "TESSERACT" in txt.upper(), f"unexpected OCR output {txt!r}"
    return repr(txt)

def ffmpeg_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "tone.wav"
        subprocess.run([config.FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        "-ar", "16000", "-ac", "1", str(out)], check=True)
        return f"generated {out.stat().st_size} B wav"

def imports():
    import fastapi, mlx_whisper, numpy, PIL, pytesseract, uvicorn, multipart  # noqa: F401,E401
    return f"fastapi {fastapi.__version__}, numpy {numpy.__version__}, pillow {PIL.__version__}"

def search_imports():
    import chromadb, sentence_transformers  # noqa: E401

    from backend import embedder
    cached = "cached" if embedder.is_cached(config.EMBED_MODEL) else "NOT downloaded yet (first sync downloads it once)"
    return f"chromadb {chromadb.__version__}, sentence-transformers {sentence_transformers.__version__}; " \
           f"{config.EMBED_MODEL} {cached}"

def embed_roundtrip():
    from backend import embedder
    e = embedder.Embedder(embedder.current_spec())
    d = e.encode_docs(["agents.md holds project rules", "bake bread at 230 C"])
    q = e.encode_query("where do coding agent rules go?")
    assert float(d[0] @ q) > float(d[1] @ q), "embedding ranking looks wrong"
    return f"dim {e.dim}, revision {e.spec['revision'][:12]}"

def graph_check():
    import mlx_lm, neo4j  # noqa: F401,E401

    from backend import embedder
    from backend.graph.store import GraphOffline, Neo4jStore
    llm = "cached" if embedder.is_cached(config.GRAPH_LLM) else "NOT downloaded"
    s = Neo4jStore()
    try:
        s.ping()
        db = f"Neo4j reachable at {config.NEO4J_URI}"
    except GraphOffline as e:
        db = f"Neo4j offline ({e}); start it with scripts/neo4j.sh start"
    finally:
        s.close()
    return f"neo4j driver {neo4j.__version__}, mlx-lm {mlx_lm.__version__}; {config.GRAPH_LLM} {llm}; {db}"

def whisper_roundtrip():
    import mlx_whisper

    with tempfile.TemporaryDirectory() as d:
        aiff, wav = Path(d) / "s.aiff", Path(d) / "s.wav"
        subprocess.run(["say", "-o", str(aiff), "The quick brown fox jumps over the lazy dog."], check=True)
        subprocess.run([config.FFMPEG, "-v", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)], check=True)
        import numpy as np
        pcm = np.frombuffer(subprocess.run([config.FFMPEG, "-v", "error", "-i", str(wav), "-f", "s16le", "-"],
                                           capture_output=True, check=True).stdout, np.int16)
        r = mlx_whisper.transcribe(pcm.astype(np.float32) / 32768.0, path_or_hf_repo=config.WHISPER_MODEL)
        assert "fox" in r["text"].lower(), r["text"]
        return repr(r["text"].strip())

print(f"python {sys.version.split()[0]} ({sys.executable})")
assert sys.version_info[:2] == (3, 11), "Python 3.11 required"
check("ffmpeg", lambda: f"{ver([config.FFMPEG, '-version'])} [{config.FFMPEG}]")
check("tesseract", lambda: f"{ver([config.TESSERACT, '--version'])} [{config.TESSERACT}]")
check("python packages", imports)
check("ffmpeg transcode", ffmpeg_roundtrip)
check("tesseract OCR", ocr_roundtrip)
check("search packages", search_imports)
if "--graph" in sys.argv:
    check("knowledge graph", graph_check)
if "--embed" in sys.argv:
    check("embedding model", embed_roundtrip)
if "--whisper" in sys.argv:
    check(f"mlx-whisper ({config.WHISPER_MODEL})", whisper_roundtrip)
sys.exit(1 if failures else 0)