"""LLM backend interface.

The agent only needs one thing from a model: turn a list of chat messages
into text. Any local model (mmap'd GGUF, HTTP server, custom runtime) can be
plugged in by implementing ``LLM.complete``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLM(ABC):
    @abstractmethod
    def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024,
                 temperature: float = 0.2, stop: Sequence[str] | None = None) -> str:
        """Return the model's reply to ``messages``."""


class CallableLLM(LLM):
    """Wrap a plain ``fn(prompt: str) -> str`` that you already have.

    Messages are flattened into a simple chat transcript. Use this when your
    model is loaded by your own code and exposes a text-in/text-out function.
    """

    def __init__(self, fn: Callable[[str], str], template: Callable[[Sequence[Message]], str] | None = None):
        self.fn = fn
        self.template = template or default_template

    def complete(self, messages, *, max_tokens=1024, temperature=0.2, stop=None):
        return self.fn(self.template(messages))


def default_template(messages: Sequence[Message]) -> str:
    parts = [f"### {m['role'].capitalize()}:\n{m['content']}\n" for m in messages]
    parts.append("### Assistant:\n")
    return "\n".join(parts)
