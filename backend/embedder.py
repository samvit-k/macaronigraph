"""Local embedding models (sentence-transformers). No network after the one-time model download.

An index is tied to an embedding *spec*: model id + resolved Hugging Face commit + dtype + normalisation +
document prompt + max sequence length + vector dimension (when fixed by the spec) + chunker version.
`signature(spec)` hashes it; a different signature means the existing vectors are incompatible and the index
must be rebuilt (indexer.py does that safely). The query instruction is recorded but does not invalidate
document vectors.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading

import numpy as np

from . import chunking, config

# Per-model query instructions (documents are embedded without a prompt).
QUERY_PROMPTS = {
    "Qwen/Qwen3-Embedding-0.6B": "Instruct: Given a question, retrieve passages from short-video transcripts, "
                                 "on-screen text and captions that answer it\nQuery: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
MAX_SEQ_LENGTH = 1024        # chunks are ≤ ~1.2k chars + header; caps memory for long captions
# `dim` is None for real models (fixed by model + revision); set explicitly where the spec chooses it.
SPEC_KEYS = ("model", "revision", "dtype", "normalize", "doc_prompt", "max_seq_length", "dim", "chunker")

def resolve_revision(model_id: str) -> str:
    """Commit hash of the locally cached snapshot (refs/main in the HF cache); downloads once if missing."""
    from pathlib import Path

    from huggingface_hub import constants, snapshot_download

    ref = Path(constants.HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    if ref.is_file() and (ref.parent.parent / "snapshots" / ref.read_text().strip()).is_dir():
        return ref.read_text().strip()
    return str(snapshot_download(model_id)).rstrip("/").split("/")[-1]

def is_cached(model_id: str) -> bool:
    from pathlib import Path

    from huggingface_hub import constants

    ref = Path(constants.HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    return ref.is_file() and (ref.parent.parent / "snapshots" / ref.read_text().strip()).is_dir()

def current_spec(model_id: str | None = None) -> dict:
    model_id = model_id or config.EMBED_MODEL
    if model_id == HashEmbedder.MODEL:
        return HashEmbedder().spec
    return {"model": model_id, "revision": resolve_revision(model_id), "dtype": config.EMBED_DTYPE,
            "normalize": True, "doc_prompt": "", "max_seq_length": MAX_SEQ_LENGTH, "dim": None,
            "chunker": chunking.CHUNKER_VERSION}

def signature(spec: dict) -> str:
    return hashlib.sha256(json.dumps({k: spec.get(k) for k in SPEC_KEYS}, sort_keys=True).encode()).hexdigest()[:16]

class Embedder:
    """sentence-transformers model, loaded lazily on first use. Loading and encoding are serialized by one
    lock, so concurrent first calls (search + indexing, or `dim`) initialise the model exactly once."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.query_prompt = QUERY_PROMPTS.get(spec["model"], "")
        self._model = None
        self._lock = threading.RLock()

    def _load(self):
        with self._lock:
            if self._model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                device = config.EMBED_DEVICE
                if device == "mps" and not torch.backends.mps.is_available():
                    device = "cpu"
                dtype = {"float32": torch.float32, "float16": torch.float16,
                         "bfloat16": torch.bfloat16}[self.spec["dtype"]]
                m = SentenceTransformer(self.spec["model"], revision=self.spec["revision"], device=device,
                                        model_kwargs={"torch_dtype": dtype}, local_files_only=True)
                # never above the model's own limit (BERT-family models have 512 positions)
                m.max_seq_length = min(self.spec["max_seq_length"], m.max_seq_length or self.spec["max_seq_length"])
                self._model = m
            return self._model

    @property
    def dim(self) -> int:
        with self._lock:
            m = self._load()
            return (getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension)()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self._load().encode(texts, batch_size=config.EMBED_BATCH, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        with self._lock:
            return self._load().encode([self.query_prompt + text], normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)

class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests (no model, no network)."""
    MODEL = "test/hash-embedder"

    def __init__(self, dim: int = 64, chunker: str | None = None):
        self.spec = {"model": self.MODEL, "revision": "v1", "dtype": "float32", "normalize": True,
                     "doc_prompt": "", "max_seq_length": 0, "dim": dim,
                     "chunker": chunker or chunking.CHUNKER_VERSION}
        self.query_prompt = ""
        self.calls = 0

    @property
    def dim(self) -> int:
        return self.spec["dim"]

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(float((v * v).sum())) or 1.0
        return v / n

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)

_cache: dict[str, object] = {}
_cache_lock = threading.Lock()

def for_spec(spec: dict):
    """One embedder instance per signature (search for an older collection keeps using its own model)."""
    sig = signature(spec)
    with _cache_lock:
        if sig not in _cache:
            _cache[sig] = HashEmbedder(dim=spec["dim"], chunker=spec["chunker"]) \
                if spec["model"] == HashEmbedder.MODEL else Embedder(spec)
        return _cache[sig]