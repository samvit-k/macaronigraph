"""Embedding specs and signatures, lazy thread-safe loading, and the real model with pinned dependencies."""
import threading
import time

import numpy as np
import pytest

from backend import config, embedder

def test_hash_embedder_dimension_is_part_of_the_signature():
    a, b = embedder.HashEmbedder(dim=64), embedder.HashEmbedder(dim=32)
    assert embedder.signature(a.spec) != embedder.signature(b.spec)
    assert embedder.signature(a.spec) == embedder.signature(embedder.HashEmbedder(dim=64).spec)

def test_for_spec_preserves_requested_dimension():
    spec = embedder.HashEmbedder(dim=24).spec
    e = embedder.for_spec(spec)
    assert e.dim == 24 and e.encode_docs(["hello world"]).shape == (1, 24)
    assert e.encode_query("hello").shape == (24,)
    assert embedder.for_spec(spec) is e
    assert embedder.for_spec(embedder.HashEmbedder(dim=48).spec).dim == 48

def test_chunker_version_is_part_of_the_signature():
    assert embedder.signature(embedder.HashEmbedder(chunker="c1").spec) != \
        embedder.signature(embedder.HashEmbedder(chunker="c2").spec)

def test_concurrent_first_use_loads_model_once(monkeypatch):
    calls = []

    class SlowST:
        def __init__(self, *a, **kw):
            calls.append(kw)
            time.sleep(0.2)
            self.max_seq_length = None

        def get_sentence_embedding_dimension(self):
            return 8

        def encode(self, texts, **kw):
            return np.ones((len(texts), 8), np.float32)

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", SlowST)
    e = embedder.Embedder({"model": "x/y", "revision": "abc", "dtype": "float32", "normalize": True,
                           "doc_prompt": "", "max_seq_length": 64, "dim": None, "chunker": "c1"})
    out = []
    ts = [threading.Thread(target=lambda: out.append(e.dim)) for _ in range(4)]
    ts += [threading.Thread(target=lambda: out.append(e.encode_docs(["a"]).shape)) for _ in range(2)]
    ts += [threading.Thread(target=lambda: out.append(e.encode_query("q").shape))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1 and calls[0]["revision"] == "abc" and calls[0]["local_files_only"] is True
    assert sorted(map(str, out)) == sorted(map(str, [8] * 4 + [(1, 8)] * 2 + [(8,)]))

@pytest.mark.skipif(not embedder.is_cached(config.EMBED_MODEL), reason="embedding model not downloaded")
def test_real_model_loads_and_encodes_with_pinned_dependencies():
    spec = embedder.current_spec()
    assert len(spec["revision"]) == 40
    e = embedder.Embedder(spec)
    docs = e.encode_docs(["Put your project rules in an agents.md file.",
                          "Bake sourdough bread at 230 degrees for 40 minutes."])
    q = e.encode_query("where do I put coding agent instructions?")
    assert docs.shape == (2, e.dim) and q.shape == (e.dim,) and docs.dtype == np.float32
    assert np.allclose(np.linalg.norm(docs, axis=1), 1, atol=1e-3)
    assert float(docs[0] @ q) > float(docs[1] @ q)

def test_max_seq_length_never_exceeds_the_models_own_limit(monkeypatch):
    class BertLike:
        def __init__(self, *a, **kw):
            self.max_seq_length = 512

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", BertLike)
    e = embedder.Embedder({"model": "x/bert", "revision": "abc", "dtype": "float32", "normalize": True,
                           "doc_prompt": "", "max_seq_length": 1024, "dim": None, "chunker": "c1"})
    assert e._load().max_seq_length == 512