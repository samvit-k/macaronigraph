"""Local LLM for graph extraction (mlx-lm on Apple Silicon): lazy loading, greedy decoding, unloadable."""
from __future__ import annotations

import gc
import threading
import time

from .. import config, embedder

class MlxLLM:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or config.GRAPH_LLM
        self.revision = (config.MODEL_REVISIONS.get(self.model_id)
                         or embedder.resolve_revision(self.model_id))
        self._model = self._tok = None
        self._lock = threading.RLock()
        self.load_seconds = 0.0

    def _load(self):
        with self._lock:
            if self._model is None:
                from huggingface_hub import snapshot_download
                from mlx_lm import load

                t0 = time.time()
                path = snapshot_download(self.model_id, revision=self.revision, local_files_only=True)
                self._model, self._tok = load(path)
                self.load_seconds += time.time() - t0
            return self._model, self._tok

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> tuple[str, dict]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        with self._lock:
            model, tok = self._load()
            prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                             add_generation_prompt=True, tokenize=False, enable_thinking=False)
            t0, parts, last = time.time(), [], None
            for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
                parts.append(r.text)
                last = r
            stats = {"seconds": round(time.time() - t0, 2)}
            if last is not None:
                stats.update(prompt_tokens=last.prompt_tokens, generation_tokens=last.generation_tokens,
                             prompt_tps=round(last.prompt_tps, 1), generation_tps=round(last.generation_tps, 1),
                             peak_memory_gb=round(last.peak_memory, 2), finish_reason=last.finish_reason)
            return "".join(parts), stats

    def unload(self) -> None:
        with self._lock:
            self._model = self._tok = None
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:  # noqa: BLE001
                pass