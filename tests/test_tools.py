from typing import Literal

from pydantic import Field

from cadtopo.schema import Tool, tool_name
from cadtopo.tools import dispatch, tool_names, tool_schema, tool_schemas


class CheckPythonSyntax(Tool):
    """Check that Python source parses."""

    action: Literal["check_python_syntax"] = "check_python_syntax"
    code: str = Field(description="The complete source.")

    def run(self) -> str:
        return "OK" if self.code.strip() else "EMPTY"


class RunPython(Tool):
    action: Literal["run_python"] = "run_python"
    code: str

    def run(self) -> str:
        return "ran"


class TestToolName:
    def test_reads_discriminator_literal(self):
        assert tool_name(CheckPythonSyntax) == "check_python_syntax"
        assert tool_name(RunPython) == "run_python"

    def test_accepts_an_instance_too(self):
        assert tool_name(CheckPythonSyntax(code="x=1")) == "check_python_syntax"

    def test_tool_names_list(self):
        assert tool_names([CheckPythonSyntax, RunPython]) == ["check_python_syntax", "run_python"]


class TestToolSchemas:
    def test_renders_native_function_descriptor(self):
        schema = tool_schema(CheckPythonSyntax)
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "check_python_syntax"
        assert fn["description"] == "Check that Python source parses."
        # The `action` discriminator is internal — never offered as an argument.
        assert "action" not in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["code"]
        assert "code" in fn["parameters"]["properties"]

    def test_tool_schemas_empty_when_no_tools(self):
        assert tool_schemas(None) == []
        assert tool_schemas([]) == []

    def test_tool_schemas_preserves_order(self):
        names = [s["function"]["name"] for s in tool_schemas([CheckPythonSyntax, RunPython])]
        assert names == ["check_python_syntax", "run_python"]


class TestDispatch:
    def test_runs_the_named_tool_from_json_arguments(self):
        assert dispatch([CheckPythonSyntax], "check_python_syntax", '{"code": "x=1"}') == "OK"

    def test_accepts_a_decoded_dict_too(self):
        assert dispatch([RunPython], "run_python", {"code": "print(1)"}) == "ran"

    def test_strips_a_stray_action_argument(self):
        out = dispatch([CheckPythonSyntax], "check_python_syntax", '{"action": "check_python_syntax", "code": "x=1"}')
        assert out == "OK"

    def test_unknown_tool_is_reported_not_raised(self):
        assert "unknown tool" in dispatch([CheckPythonSyntax], "nope", "{}")

    def test_malformed_json_is_reported_not_raised(self):
        assert "invalid arguments" in dispatch([CheckPythonSyntax], "check_python_syntax", "{bad")

    def test_missing_required_argument_is_reported_not_raised(self):
        assert "tool error" in dispatch([CheckPythonSyntax], "check_python_syntax", "{}")

    def test_a_raising_tool_is_captured_as_an_error_string(self):
        class Broken(Tool):
            action: Literal["boom"] = "boom"

            def run(self) -> str:
                raise RuntimeError("tool exploded")

        assert "tool exploded" in dispatch([Broken], "boom", "{}")
