import json

import pytest

from agent import Agent, CallableLLM, Policy
from agent.brain import parse_decision
from agent.llm.base import LLM


class ScriptedLLM(LLM):
    """Replays canned replies and records the messages it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, **kw):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


def act(tool, **args):
    return json.dumps({"thought": f"use {tool}", "action": {"tool": tool, "args": args}})


def final(text):
    return json.dumps({"thought": "done", "final": text})


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "notes.txt").write_text("the secret is 42")
    return tmp_path


def test_sense_then_answer(ws):
    llm = ScriptedLLM([act("list_dir", path="."), act("read_file", path="notes.txt"), final("42")])
    state = Agent(llm, policy=Policy(workspace=ws)).run("find the secret")
    assert state.final == "42"
    assert [s.tool for s in state.steps] == ["list_dir", "read_file"]
    assert "notes.txt" in state.steps[0].observation
    # the observation is fed back to the brain on the next step
    assert "the secret is 42" in llm.calls[2][-1]["content"]


def test_actuator_writes_inside_workspace(ws):
    llm = ScriptedLLM([act("write_file", path="out/a.txt", content="hi"), final("ok")])
    Agent(llm, policy=Policy(workspace=ws)).run("write")
    assert (ws / "out/a.txt").read_text() == "hi"


def test_workspace_escape_is_blocked(ws):
    llm = ScriptedLLM([act("read_file", path="../../etc/passwd"), final("no")])
    state = Agent(llm, policy=Policy(workspace=ws)).run("escape")
    assert not state.steps[0].ok
    assert "outside the workspace" in state.steps[0].observation


def test_shell_and_network_off_by_default(ws):
    llm = ScriptedLLM([act("run_shell", command="echo hi"), act("http_get", url="http://x"), final("x")])
    state = Agent(llm, policy=Policy(workspace=ws)).run("go")
    assert "disabled" in state.steps[0].observation
    assert "disabled" in state.steps[1].observation


def test_shell_when_allowed(ws):
    llm = ScriptedLLM([act("run_shell", command="echo hi"), final("x")])
    state = Agent(llm, policy=Policy(workspace=ws, allow_shell=True)).run("go")
    assert "exit=0" in state.steps[0].observation and "hi" in state.steps[0].observation


def test_operator_can_block_actuator(ws):
    llm = ScriptedLLM([act("write_file", path="a.txt", content="x"), final("x")])
    Agent(llm, policy=Policy(workspace=ws, confirm=lambda t, a: False)).run("go")
    assert not (ws / "a.txt").exists()


def test_parse_error_is_fed_back(ws):
    llm = ScriptedLLM(["I think I should look around.", final("done")])
    state = Agent(llm, policy=Policy(workspace=ws)).run("go")
    assert not state.steps[0].ok
    assert "JSON" in llm.calls[1][-1]["content"]
    assert state.final == "done"


def test_repeat_action_is_flagged(ws):
    llm = ScriptedLLM([act("list_dir", path="."), act("list_dir", path="."), final("x")])
    state = Agent(llm, policy=Policy(workspace=ws)).run("go")
    assert state.steps[0].ok and not state.steps[1].ok


def test_unknown_tool_and_bad_args(ws):
    llm = ScriptedLLM([act("fly"), act("read_file", nope="x"), final("x")])
    state = Agent(llm, policy=Policy(workspace=ws)).run("go")
    assert "unknown tool" in state.steps[0].observation
    assert "unexpected args" in state.steps[1].observation


def test_stops_on_budget(ws):
    llm = ScriptedLLM([act("list_dir", path=str(i)) for i in range(5)])
    state = Agent(llm, policy=Policy(workspace=ws), max_steps=3).run("go")
    assert state.final is None and state.stop_reason == "step budget exhausted"


def test_stops_on_consecutive_errors(ws):
    llm = ScriptedLLM(["junk"] * 10)
    state = Agent(llm, policy=Policy(workspace=ws)).run("go")
    assert "consecutive errors" in state.stop_reason
    assert state.step_count == 4


def test_old_steps_are_summarized(ws):
    replies = [act("list_dir", path="."), act("system_info"), act("read_file", path="notes.txt"), final("x")]
    llm = ScriptedLLM(replies)
    Agent(llm, policy=Policy(workspace=ws), window=1).run("go")
    last = llm.calls[-1]
    assert "Earlier steps (summarized)" in last[1]["content"]
    assert len(last) == 4  # system, goal, one replayed assistant + observation


def test_callable_llm_wraps_plain_function(ws):
    llm = CallableLLM(lambda prompt: final("from fn") if "Goal: go" in prompt else "")
    assert Agent(llm, policy=Policy(workspace=ws)).run("go").final == "from fn"


@pytest.mark.parametrize("raw", [
    'Sure!\n```json\n{"thought": "t", "final": "a"}\n```',
    'blah {"thought": "t", "final": "a"} trailing',
])
def test_parse_tolerates_chatter(raw):
    assert parse_decision(raw).final == "a"
