"""Command line entry point: ``python -m agent "your goal"``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .body import Agent
from .feedback import Step
from .llm import CallableLLM, HttpLLM, LlamaCppLLM
from .tools import Policy


def load_llm(args):
    if args.model:
        return LlamaCppLLM(args.model, n_ctx=args.ctx, n_gpu_layers=args.gpu_layers)
    if args.fn:
        module, _, attr = args.fn.partition(":")
        return CallableLLM(getattr(importlib.import_module(module), attr))
    return HttpLLM(args.url, model=args.model_name)


def confirm(tool: str, args: dict) -> bool:
    print(f"\n[actuator] {tool} {json.dumps(args)[:500]}", file=sys.stderr)
    return input("allow? [y/N] ").strip().lower() in ("y", "yes")


def print_step(step: Step) -> None:
    status = "ok" if step.ok else "ERROR"
    print(f"\n--- step {step.index} ---", file=sys.stderr)
    if step.thought:
        print(f"thought: {step.thought}", file=sys.stderr)
    print(f"action:  {step.tool} {json.dumps(step.args)[:300]}", file=sys.stderr)
    print(f"{status}:      {step.observation[:600]}", file=sys.stderr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agent", description="Run the agent against a goal.")
    p.add_argument("goal")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--model", help="path to a GGUF file (loaded with mmap via llama-cpp-python)")
    src.add_argument("--fn", help="module:function of your own generate(prompt) -> str")
    src.add_argument("--url", default="http://127.0.0.1:8080/v1",
                     help="OpenAI-compatible endpoint (default: llama-server on :8080)")
    p.add_argument("--model-name", default="local")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--gpu-layers", type=int, default=0)
    p.add_argument("--workspace", default=".", help="directory the tools are confined to")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--allow-shell", action="store_true")
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--read-only", action="store_true", help="disable file writes")
    p.add_argument("--yes", action="store_true", help="run actuators without asking")
    p.add_argument("--json", action="store_true", help="print the final state vector as JSON")
    args = p.parse_args(argv)

    policy = Policy(workspace=Path(args.workspace), allow_shell=args.allow_shell,
                    allow_network=args.allow_network, allow_write=not args.read_only,
                    confirm=None if args.yes else confirm)
    agent = Agent(load_llm(args), policy=policy, max_steps=args.max_steps, on_step=print_step)
    state = agent.run(args.goal)

    if args.json:
        print(json.dumps(state.to_dict(), indent=2))
    else:
        print(f"\n[{state.stop_reason}]", file=sys.stderr)
        print(state.final or "(no final answer)")
    return 0 if state.final is not None else 1


if __name__ == "__main__":
    sys.exit(main())
