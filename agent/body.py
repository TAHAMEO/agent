"""II. Agent Body: the orchestrator.

Runs the loop from the architecture diagram:

    Core Brain --decision--> Agent Body --tool call--> Toolset --> OS / Network
        ^                                                 |
        +---------- Feedback Loop (state vector) <--------+
"""

from __future__ import annotations

from typing import Callable

from .brain import Brain, Decision
from .feedback import StateVector, Step
from .llm.base import LLM
from .tools import Policy, ToolError, Toolset, default_toolset


class Agent:
    def __init__(self, llm: LLM, toolset: Toolset | None = None, *, policy: Policy | None = None,
                 max_steps: int = 20, window: int = 8,
                 on_step: Callable[[Step], None] | None = None):
        self.brain = Brain(llm)
        self.toolset = toolset or default_toolset(policy)
        self.max_steps = max_steps
        self.window = window
        self.on_step = on_step

    def run(self, goal: str) -> StateVector:
        state = StateVector(goal=goal, max_steps=self.max_steps, window=self.window)
        tools = self.toolset.describe()
        while not state.done:
            decision = self.brain.decide(state, tools)
            if decision.final is not None:
                state.finish(decision.final)
                break
            self._record(state, decision, *self._act(state, decision))
        return state

    def _act(self, state: StateVector, d: Decision) -> tuple[str, bool]:
        if d.error:
            return d.error, False
        if state.is_repeat(d.tool, d.args):
            return ("you just ran this exact action; use its observation above "
                    "or try something different"), False
        try:
            return self.toolset.call(d.tool, d.args), True
        except ToolError as e:
            return str(e), False

    def _record(self, state: StateVector, d: Decision, observation: str, ok: bool) -> None:
        step = Step(index=state.step_count + 1, thought=d.thought, tool=d.tool, args=d.args,
                    observation=observation, ok=ok, raw=d.raw)
        state.record(step)
        if self.on_step:
            self.on_step(step)
