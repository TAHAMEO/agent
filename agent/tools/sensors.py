"""Sensors: read-only tools that observe the OS and network."""

from __future__ import annotations

import os
import platform
import shutil
import urllib.request

from .base import Policy, ToolError, Toolset


def list_dir(path: str = ".", *, policy: Policy) -> str:
    """List the entries of a directory in the workspace."""
    p = policy.resolve(path)
    if not p.is_dir():
        raise ToolError(f"{path} is not a directory")
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(empty)"


def read_file(path: str, *, policy: Policy) -> str:
    """Read a text file from the workspace."""
    p = policy.resolve(path)
    if not p.is_file():
        raise ToolError(f"{path} is not a file")
    return p.read_text(errors="replace")


def search_files(pattern: str, path: str = ".", *, policy: Policy) -> str:
    """Find files under a directory whose name matches a glob pattern."""
    root = policy.resolve(path)
    hits = [str(p.relative_to(policy.workspace.resolve())) for p in root.rglob(pattern)]
    return "\n".join(sorted(hits)[:200]) or "(no matches)"


def system_info(*, policy: Policy) -> str:
    """Report OS, CPU, and disk details of the host."""
    usage = shutil.disk_usage(policy.workspace)
    return (f"os={platform.system()} {platform.release()} arch={platform.machine()} "
            f"python={platform.python_version()} cpus={os.cpu_count()} "
            f"disk_free_gb={usage.free / 1e9:.1f} workspace={policy.workspace.resolve()}")


def http_get(url: str, *, policy: Policy) -> str:
    """Fetch a URL over HTTP(S) and return the response body."""
    if not policy.allow_network:
        raise ToolError("network access is disabled by policy")
    if not url.startswith(("http://", "https://")):
        raise ToolError("only http(s) URLs are allowed")
    with urllib.request.urlopen(url, timeout=30) as resp:
        return f"status={resp.status}\n" + resp.read(policy.max_output_chars * 2).decode(errors="replace")


def register(toolset: Toolset) -> None:
    for fn in (list_dir, read_file, search_files, system_info, http_get):
        toolset.sensor(fn)
