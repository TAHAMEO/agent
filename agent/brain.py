"""I. Core Brain: LLM reasoning.

Turns the state vector into a prompt, asks the model what to do next, and
parses its reply into a ``Decision``. A plain JSON protocol is used instead
of native function calling, so it works with any local model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .feedback import StateVector
from .llm.base import LLM, Message

SYSTEM_PROMPT = """You are an autonomous agent. You reach the user's goal by reasoning step by step and calling tools.

Tools (sensors observe, actuators act):
{tools}

Reply with exactly ONE JSON object and nothing else, in one of these two forms:
{{"thought": "<brief reasoning>", "action": {{"tool": "<tool name>", "args": {{...}}}}}}
{{"thought": "<brief reasoning>", "final": "<answer for the user>"}}

Rules:
- One action per reply. Wait for its observation before the next one.
- Use only the tools listed above, with the argument names shown.
- If an action fails, read the error and try something different.
- When the goal is reached or cannot be reached, reply with "final"."""


@dataclass
class Decision:
    thought: str = ""
    tool: str | None = None
    args: dict = field(default_factory=dict)
    final: str | None = None
    error: str | None = None  # set when the reply could not be parsed
    raw: str = ""


class Brain:
    def __init__(self, llm: LLM, *, max_tokens: int = 1024, temperature: float = 0.2):
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature

    def build_messages(self, state: StateVector, tools_description: str) -> list[Message]:
        user = f"Goal: {state.goal}"
        earlier = state.summary()
        if earlier:
            user += f"\n\nEarlier steps (summarized):\n{earlier}"
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools_description)},
            {"role": "user", "content": user},
        ]
        for s in state.recent():
            messages.append({"role": "assistant", "content": s.raw})
            label = "Observation" if s.ok else "Error"
            messages.append({"role": "user", "content": f"{label}:\n{s.observation}"})
        messages[-1]["content"] += "\n\n[state: " + "; ".join(state.signals()) + "]"
        return messages

    def decide(self, state: StateVector, tools_description: str) -> Decision:
        messages = self.build_messages(state, tools_description)
        raw = self.llm.complete(messages, max_tokens=self.max_tokens, temperature=self.temperature)
        return parse_decision(raw)


def parse_decision(raw: str) -> Decision:
    obj = _extract_json(raw)
    if obj is None:
        return Decision(raw=raw, error="reply was not a JSON object; reply with exactly one JSON object")
    thought = str(obj.get("thought", ""))
    if "final" in obj:
        return Decision(thought=thought, final=str(obj["final"]), raw=raw)
    action = obj.get("action")
    if isinstance(action, dict) and isinstance(action.get("tool"), str):
        args = action.get("args") or {}
        if not isinstance(args, dict):
            return Decision(thought=thought, raw=raw, error="action.args must be a JSON object")
        return Decision(thought=thought, tool=action["tool"], args=args, raw=raw)
    return Decision(thought=thought, raw=raw,
                    error='JSON must contain either "action": {"tool", "args"} or "final"')


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Find the first JSON object in ``text``, tolerating code fences and chatter."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        for i, ch in enumerate(chunk):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(chunk, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None
