"""Local GGUF model through llama-cpp-python, memory-mapped from disk."""

from __future__ import annotations

from .base import LLM


class LlamaCppLLM(LLM):
    def __init__(self, model_path: str | None = None, *, llama=None, n_ctx: int = 8192,
                 n_gpu_layers: int = 0, use_mmap: bool = True, **kwargs):
        """Either pass ``model_path`` to load a GGUF file (mmap'd by default),
        or pass an already-loaded ``llama_cpp.Llama`` instance as ``llama``."""
        if llama is None:
            if model_path is None:
                raise ValueError("model_path or llama is required")
            from llama_cpp import Llama  # optional dependency

            llama = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                          use_mmap=use_mmap, verbose=False, **kwargs)
        self.llama = llama

    def complete(self, messages, *, max_tokens=1024, temperature=0.2, stop=None):
        out = self.llama.create_chat_completion(
            messages=list(messages), max_tokens=max_tokens,
            temperature=temperature, stop=list(stop) if stop else None,
        )
        return out["choices"][0]["message"]["content"] or ""
