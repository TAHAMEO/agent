"""V. Feedback Loop: the state vector fed back into the Core Brain each step.

It records what the agent did and what the toolset observed, detects loops
and repeated failures, and renders all of it as context for the next
reasoning step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Step:
    index: int
    thought: str
    tool: str | None
    args: dict
    observation: str
    ok: bool
    raw: str  # the model's raw output, replayed as the assistant turn


@dataclass
class StateVector:
    goal: str
    max_steps: int = 20
    window: int = 8  # most recent steps replayed verbatim; older ones are summarized
    max_consecutive_errors: int = 4
    steps: list[Step] = field(default_factory=list)
    consecutive_errors: int = 0
    done: bool = False
    final: str | None = None
    stop_reason: str | None = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_left(self) -> int:
        return self.max_steps - self.step_count

    def record(self, step: Step) -> None:
        self.steps.append(step)
        self.consecutive_errors = 0 if step.ok else self.consecutive_errors + 1
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.stop(f"stopped after {self.consecutive_errors} consecutive errors")
        elif self.budget_left <= 0:
            self.stop("step budget exhausted")

    def finish(self, answer: str) -> None:
        self.done, self.final, self.stop_reason = True, answer, "finished"

    def stop(self, reason: str) -> None:
        self.done, self.stop_reason = True, reason

    def is_repeat(self, tool: str | None, args: dict) -> bool:
        """True when the same action was just tried with the same arguments."""
        if not self.steps or tool is None:
            return False
        last = self.steps[-1]
        return last.tool == tool and last.args == args

    def signals(self) -> list[str]:
        """Feedback signals that steer the next reasoning step."""
        out = [f"step {self.step_count + 1} of {self.max_steps}"]
        if self.consecutive_errors:
            out.append(f"{self.consecutive_errors} consecutive error(s): change approach")
        if self.budget_left <= 2:
            out.append("budget nearly exhausted: give a final answer soon")
        return out

    def summary(self) -> str:
        """One line per step that has fallen out of the replay window."""
        older = self.steps[:-self.window] if self.window else self.steps
        return "\n".join(
            f"#{s.index} {s.tool}({json.dumps(s.args)[:120]}) -> "
            f"{'ok' if s.ok else 'ERROR'}: {s.observation[:160].replace(chr(10), ' ')}"
            for s in older
        )

    def recent(self) -> list[Step]:
        return self.steps[-self.window:] if self.window else []

    def to_dict(self) -> dict:
        return {
            "goal": self.goal, "done": self.done, "final": self.final,
            "stop_reason": self.stop_reason, "steps": [
                {"index": s.index, "thought": s.thought, "tool": s.tool, "args": s.args,
                 "ok": s.ok, "observation": s.observation} for s in self.steps
            ],
        }
