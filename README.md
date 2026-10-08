# agent

A tool-using agent that runs on top of **your own local LLM** (for example a GGUF
model memory-mapped with llama.cpp). It has no required dependencies, only the
Python standard library.

## Architecture

```
+---------------------------------------------------------------+
|   +---------------------+       +--------------------------+  |
|   | I. Core Brain       |       | II. Agent Body           |  |
|   | LLM Reasoning       | ----> | Orchestrator / Tool Call |  |
|   +---------------------+       +--------------------------+  |
|              ^                              |                 |
|              |                              v                 |
|   +---------------------+       +--------------------------+  |
|   | V. Feedback Loop    | <---- | III. Toolset             |  |
|   | State Vector        |       | Sensors & Actuators      |  |
|   +---------------------+       +--------------------------+  |
|              ^                              |                 |
+--------------|------------------------------|-----------------+
               |                              v
        +-------------------------------------------+
        |        Operating System / Network         |
        +-------------------------------------------+
```

| Layer | Module | What it does |
|---|---|---|
| I. Core Brain | `agent/brain.py`, `agent/llm/` | Builds the prompt from the state vector, calls your LLM, and parses the reply into a `Decision` (a tool call or a final answer). It uses a plain JSON protocol, so it works with any model, even without native function calling. |
| II. Agent Body | `agent/body.py` | The orchestrator loop: decide, run the tool, record the result, repeat until a final answer or a stop condition. |
| III. Toolset | `agent/tools/` | **Sensors** (`list_dir`, `read_file`, `search_files`, `system_info`, `http_get`) and **actuators** (`write_file`, `run_shell`). |
| V. Feedback Loop | `agent/feedback.py` | The `StateVector`: the goal, the step history, observations, error streaks, and the step budget. It replays recent steps, summarizes older ones, flags repeated actions, and stops on budget exhaustion or too many errors in a row. |
| OS / Network | `Policy` in `agent/tools/base.py` | Confines file access to a workspace directory. Shell and network access are **off** by default, and every actuator call goes through a confirmation hook. |

## Plug in your LLM

Pick whichever matches how your model is already loaded:

```python
from agent import Agent, Policy, LlamaCppLLM, HttpLLM, CallableLLM

# 1) GGUF file, memory-mapped by llama-cpp-python (pip install ".[llamacpp]")
llm = LlamaCppLLM("/models/my-model.gguf", n_ctx=8192, n_gpu_layers=0)  # use_mmap=True by default
#    ...or reuse a Llama instance you already loaded:
llm = LlamaCppLLM(llama=my_llama)

# 2) A running llama-server / any OpenAI-compatible endpoint
llm = HttpLLM("http://127.0.0.1:8080/v1")

# 3) Your own runtime: any function prompt -> text
llm = CallableLLM(my_generate_fn)

agent = Agent(llm, policy=Policy(workspace="./sandbox", allow_shell=False))
state = agent.run("Summarize the README files in this directory")
print(state.final)
```

To use a custom runtime with chat-style input, subclass `agent.llm.LLM` and implement
`complete(messages, max_tokens, temperature, stop) -> str`.

## CLI

```bash
pip install -e .

agent "list the python files and summarize them" --model /models/my-model.gguf
agent "check disk space" --url http://127.0.0.1:8080/v1
agent "..." --fn mypkg.runtime:generate --workspace ./sandbox --allow-shell
```

By default the CLI asks before every actuator call. Pass `--yes` to skip the prompts
and `--json` to print the full state vector.

## Adding tools

```python
from agent import default_toolset, Policy

tools = default_toolset(Policy(workspace="."))

@tools.sensor
def cpu_temp() -> str:
    """Read the CPU temperature."""
    return open("/sys/class/thermal/thermal_zone0/temp").read()

@tools.actuator
def notify(message: str) -> str:
    """Send a desktop notification."""
    ...

agent = Agent(llm, toolset=tools)
```

The tool's signature and the first line of its docstring become its description in
the prompt. Add a `policy` keyword parameter to receive the active `Policy`.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
