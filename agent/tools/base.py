"""III. Toolset: the agent's sensors (observe) and actuators (act).

Every tool touches the Operating System / Network layer, so each call is
checked against a ``Policy`` before it runs.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class ToolError(Exception):
    """Raised by a tool to report a failure back to the model."""


@dataclass
class Policy:
    """Guards between the toolset and the OS / network."""

    workspace: Path = field(default_factory=Path.cwd)
    allow_shell: bool = False
    allow_network: bool = False
    allow_write: bool = True
    # Called before any actuator runs; return False to block it.
    confirm: Callable[[str, dict], bool] | None = None
    max_output_chars: int = 8000

    def resolve(self, path: str) -> Path:
        """Resolve ``path`` inside the workspace, refusing to escape it."""
        root = self.workspace.resolve()
        p = (root / path).resolve()
        if p != root and root not in p.parents:
            raise ToolError(f"path {path!r} is outside the workspace {root}")
        return p


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    kind: str  # "sensor" | "actuator"
    params: dict[str, str]

    def signature(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.params.items())
        return f"{self.name}({args}) [{self.kind}] - {self.description}"


class Toolset:
    def __init__(self, policy: Policy | None = None):
        self.policy = policy or Policy()
        self.tools: dict[str, Tool] = {}

    def register(self, fn: Callable[..., Any], *, kind: str, name: str | None = None,
                 description: str | None = None) -> Tool:
        sig = inspect.signature(fn)
        params = {
            p.name: getattr(p.annotation, "__name__", str(p.annotation))
            for p in sig.parameters.values() if p.name != "policy"
        }
        tool = Tool(name=name or fn.__name__,
                    description=description or (inspect.getdoc(fn) or "").split("\n")[0],
                    fn=fn, kind=kind, params=params)
        self.tools[tool.name] = tool
        return tool

    def sensor(self, fn=None, **kw):
        if fn is None:
            return lambda f: self.sensor(f, **kw)
        self.register(fn, kind="sensor", **kw)
        return fn

    def actuator(self, fn=None, **kw):
        if fn is None:
            return lambda f: self.actuator(f, **kw)
        self.register(fn, kind="actuator", **kw)
        return fn

    def describe(self) -> str:
        return "\n".join(f"- {t.signature()}" for t in self.tools.values())

    def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.tools)}")
        if not isinstance(args, dict):
            raise ToolError("args must be a JSON object")
        unknown = set(args) - set(tool.params)
        if unknown:
            raise ToolError(f"unexpected args for {name}: {sorted(unknown)}")
        if tool.kind == "actuator" and self.policy.confirm and not self.policy.confirm(name, args):
            raise ToolError(f"{name} was blocked by the operator")
        kwargs = dict(args)
        if "policy" in inspect.signature(tool.fn).parameters:
            kwargs["policy"] = self.policy
        try:
            result = tool.fn(**kwargs)
        except ToolError:
            raise
        except TypeError as e:
            raise ToolError(f"bad arguments for {name}: {e}") from e
        except Exception as e:  # surface OS / network failures to the model
            raise ToolError(f"{type(e).__name__}: {e}") from e
        text = result if isinstance(result, str) else repr(result)
        limit = self.policy.max_output_chars
        if len(text) > limit:
            text = text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
        return text
