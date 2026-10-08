"""Actuators: tools that change the state of the OS."""

from __future__ import annotations

import subprocess

from .base import Policy, ToolError, Toolset


def write_file(path: str, content: str, *, policy: Policy) -> str:
    """Create or overwrite a text file in the workspace."""
    if not policy.allow_write:
        raise ToolError("file writes are disabled by policy")
    p = policy.resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def run_shell(command: str, *, policy: Policy) -> str:
    """Run a shell command in the workspace and return its exit code and output."""
    if not policy.allow_shell:
        raise ToolError("shell access is disabled by policy")
    proc = subprocess.run(command, shell=True, cwd=policy.workspace, capture_output=True,
                          text=True, timeout=120)
    return f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def register(toolset: Toolset) -> None:
    for fn in (write_file, run_shell):
        toolset.actuator(fn)
