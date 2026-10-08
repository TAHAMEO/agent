"""OpenAI-compatible chat endpoint (llama.cpp ``llama-server``, vLLM, Ollama, ...)."""

from __future__ import annotations

import json
import urllib.request

from .base import LLM


class HttpLLM(LLM):
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", model: str = "local",
                 api_key: str | None = None, timeout: float = 300):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, messages, *, max_tokens=1024, temperature=0.2, stop=None):
        body = {"model": self.model, "messages": list(messages),
                "max_tokens": max_tokens, "temperature": temperature}
        if stop:
            body["stop"] = list(stop)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"] or ""
