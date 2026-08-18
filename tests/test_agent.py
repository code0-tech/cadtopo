from types import SimpleNamespace
from typing import Literal

from cadtopo.agent import Agent
from cadtopo.schema import Descriptors, Tool


class CheckPythonSyntax(Tool):
    action: Literal["check_python_syntax"] = "check_python_syntax"
    code: str

    def run(self) -> str:
        return "OK: no syntax errors."


def _agent(**overrides) -> Agent:
    defaults = dict(
        name="Developer",
        skill_definition="Implement code.",
        cost_per_token=1.0,
        api_provider="test-model",
        api_key="key",
        system_prompt="You are Developer.",
    )
    defaults.update(overrides)
    return Agent(**defaults)


def _descriptors(query="q", key="k", failure_risk=0.3) -> Descriptors:
    return Descriptors(query=query, key=key, failure_risk=failure_risk)


def _tool_call(name, arguments, id="call_1"):
    """A provider-shaped native tool call (``message.tool_calls[i]``)."""
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=arguments))


def _completion(content=None, tool_calls=None):
    """A minimal LiteLLM/OpenAI ``ChatCompletion`` (native turn or plain text)."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def _tool_agent(**overrides) -> Agent:
    return _agent(tools=[CheckPythonSyntax], **overrides)


class TestStepMessageAssembly:
    def test_round_1_omits_round_goal_message(self, scripted_llm):
        scripted = scripted_llm([_completion(content="ans")])
        _agent().step(task="Reverse a string.", round_goal="", history=[])

        messages = scripted.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "Task (constant across all rounds):\n\nReverse a string."}
        assert not any("This round's goal" in m.get("content", "") for m in messages)

    def test_later_round_appends_goal_after_history(self, scripted_llm):
        scripted = scripted_llm([_completion(content="ans")])
        history = [{"role": "assistant", "content": "prior public message"}]
        _agent().step(task="Reverse a string.", round_goal="Fix the edge case.", history=history)

        messages = scripted.calls[0]["messages"]
        assert messages[2] == history[0]
        assert "Fix the edge case." in messages[-1]["content"]
        assert messages[-1]["role"] == "user"

    def test_system_prompt_and_guidance_both_present(self, scripted_llm):
        scripted = scripted_llm([_completion(content="ans")])
        _agent(system_prompt="ROLE TEXT").step(task="t", round_goal="", history=[])
        system_msg = scripted.calls[0]["messages"][0]["content"]
        assert "ROLE TEXT" in system_msg
        assert "deliverable" in system_msg  # the step guidance


class TestDescribe:
    def test_returns_only_descriptors_and_makes_one_call(self, scripted_structured):
        scripted = scripted_structured([_descriptors(query="need the candidate code", key="I can execute it", failure_risk=0.4)])
        step = _agent().describe(task="Reverse a string.", round_goal="", history=[])

        assert len(scripted.calls) == 1
        assert scripted.calls[0]["response_model"] is Descriptors
        assert step.query == "need the candidate code"
        assert step.key == "I can execute it"
        # Confidence is the inverted self-report: 1 - failure_risk.
        assert step.accuracy == 0.6
        # Phase 1 is descriptors only — no work carried out of describe().
        assert step.public == ""

    def test_describe_prompt_forbids_work_and_omits_step_guidance(self, scripted_structured):
        scripted = scripted_structured([_descriptors()])
        _tool_agent().describe(task="t", round_goal="", history=[])
        system_msg = scripted.calls[0]["messages"][0]["content"]
        assert "declare ONLY your routing descriptors" in system_msg
        assert "deliverable" not in system_msg  # no work-format guidance in the descriptor call

    def test_describe_failure_falls_open_to_neutral_descriptors(self, scripted_structured):
        scripted_structured([ConnectionError("down")])
        step = _agent().describe("t", "", [])
        assert step.query == ""
        assert step.key == ""
        assert step.accuracy == 0.5


class TestStepNoTools:
    def test_returns_the_plain_text_deliverable(self, scripted_llm):
        scripted = scripted_llm([_completion(content="42")])
        step = _agent().step("t", "", [])
        assert len(scripted.calls) == 1
        assert step.public == "42"
        assert step.tool_calls == []

    def test_transport_failure_returns_error_step_without_raising(self, scripted_llm):
        scripted_llm([ConnectionError("boom")])
        step = _agent().step("t", "", [])
        assert step.accuracy == 0.0
        assert "ERROR" in step.public
        assert "Developer" in step.public


class TestToolCalling:
    """Tools are offered natively; once the model stops calling them, its
    plain-text reply is the deliverable — no separate structured call.

    Every turn is a native ``litellm.completion`` (patched by ``scripted_llm``).
    """

    def test_tool_call_then_plain_text_deliverable(self, scripted_llm):
        native = scripted_llm([
            _completion(tool_calls=[_tool_call("check_python_syntax", '{"code": "x=1"}')]),
            _completion(content="done"),  # no tool calls -> this content IS the deliverable
        ])
        step = _tool_agent().step("t", "", [])

        assert step.public == "done"
        assert len(step.tool_calls) == 1
        assert step.tool_calls[0]["name"] == "check_python_syntax"
        assert step.tool_calls[0]["result"] == "OK: no syntax errors."
        assert len(native.calls) == 2

    def test_immediate_deliverable_when_no_tool_is_called(self, scripted_llm):
        native = scripted_llm([_completion(content="I can answer directly")])
        step = _tool_agent().step("t", "", [])

        assert len(native.calls) == 1
        assert step.public == "I can answer directly"
        assert step.tool_calls == []

    def test_duplicate_tool_call_in_same_pass_is_served_from_cache(self, scripted_llm):
        calls_made = []

        class Counting(Tool):
            action: Literal["check_python_syntax"] = "check_python_syntax"
            code: str

            def run(self) -> str:
                calls_made.append(self.code)
                return "OK"

        scripted_llm([
            _completion(tool_calls=[_tool_call("check_python_syntax", '{"code": "x=1"}')]),
            _completion(tool_calls=[_tool_call("check_python_syntax", '{"code": "x=1"}')]),
            _completion(content="done"),
        ])
        step = _agent(tools=[Counting]).step("t", "", [])

        assert len(calls_made) == 1          # de-duped: identical (name, arguments)
        assert len(step.tool_calls) == 2     # but both invocations are still logged
        assert step.public == "done"

    def test_broken_tool_reports_error_without_killing_the_pass(self, scripted_llm):
        class Broken(Tool):
            action: Literal["check_python_syntax"] = "check_python_syntax"
            code: str

            def run(self) -> str:
                raise RuntimeError("tool exploded")

        scripted_llm([
            _completion(tool_calls=[_tool_call("check_python_syntax", '{"code": "x=1"}')]),
            _completion(content="done"),
        ])
        step = _agent(tools=[Broken]).step("t", "", [])

        assert "tool exploded" in step.tool_calls[0]["result"]
        assert step.public == "done"

    def test_iteration_cap_forces_a_final_plain_text_call(self, scripted_llm):
        agent = _tool_agent()
        agent.max_tool_iterations = 2
        native = scripted_llm([
            _completion(tool_calls=[_tool_call("check_python_syntax", '{"code": "x=1"}')]),
            _completion(tool_calls=[_tool_call("check_python_syntax", '{"code": "y=2"}')]),
            _completion(content="forced"),  # the final plain-text completion after the cap
        ])
        step = agent.step("t", "", [])

        assert len(native.calls) == 3  # 2 tool turns + 1 forced final text call
        assert "Do NOT call any more tools" in native.calls[-1]["messages"][-1]["content"]
        assert step.public == "forced"
        assert len(step.tool_calls) == 2

    def test_tool_result_is_fed_back_into_the_conversation(self, scripted_llm):
        native = scripted_llm([
            _completion(tool_calls=[_tool_call("check_python_syntax", '{"code": "x=1"}')]),
            _completion(content="done"),
        ])
        _tool_agent().step("t", "", [])

        # The second native turn's messages must carry the tool result back.
        second = native.calls[1]["messages"]
        assert any(m.get("role") == "tool" and "OK: no syntax errors." in m.get("content", "") for m in second)
        # ...preceded by the assistant turn that requested it.
        assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second)
