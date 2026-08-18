"""Tool wiring for the native tool-calling protocol.

A tool is a :class:`cadtopo.schema.Tool` subclass — arguments as pydantic
fields, a discriminator ``action`` literal, and a :meth:`run` method. Tools are
handed to the provider through the STANDARD tool-calling interface every
provider supports: :func:`tool_schemas` renders each tool into the OpenAI-style
``{"type": "function", "function": {...}}` descriptor passed as ``tools=[...]``
on the completion call (see :func:`cadtopo.llm.tool_completion`). The model then
selects and fills a tool through the provider's own machinery; once it stops
calling tools, its plain-text reply is the agent's deliverable (no schema).

Between turns the harness executes any tool the model called: :func:`dispatch`
maps a returned tool-call (name + JSON arguments) back to its ``Tool`` subclass,
validates the arguments with pydantic, and runs it — a bad name, malformed
arguments, or a raising :meth:`run` all come back as a ``[tool error: …]``
string rather than killing the round.
"""

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import ValidationError

from .schema import Tool, tool_name


def _parameters_schema(tool: Type[Tool]) -> Dict[str, Any]:
    """The tool's JSON-Schema arguments, with the ``action`` discriminator dropped.

    ``action`` is an internal routing literal, not a model-facing argument, so
    it is stripped from both ``properties`` and ``required``.
    """
    schema = tool.model_json_schema()
    props = {k: v for k, v in schema.get("properties", {}).items() if k != "action"}
    required = [r for r in schema.get("required", []) if r != "action"]
    out: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


def tool_schema(tool: Type[Tool]) -> Dict[str, Any]:
    """One tool rendered as the provider's native function descriptor."""
    return {
        "type": "function",
        "function": {
            "name": tool_name(tool),
            "description": (tool.__doc__ or "").strip(),
            "parameters": _parameters_schema(tool),
        },
    }


def tool_schemas(tools: Optional[List[Type[Tool]]]) -> List[Dict[str, Any]]:
    """The ``tools=[...]`` payload for a completion call (empty if no tools)."""
    return [tool_schema(t) for t in tools or []]


def tool_names(tools: Optional[List[Type[Tool]]]) -> List[str]:
    """The discriminator names of a tool list (order preserved)."""
    return [tool_name(t) for t in tools or []]


def dispatch(tools: Optional[List[Type[Tool]]], name: str, arguments: Any) -> str:
    """Run the tool the model called, returning its result (or a ``[tool error]``).

    ``arguments`` is whatever the provider handed back for the call — usually a
    JSON string, occasionally a decoded dict. A bad name, unparseable arguments,
    schema-invalid arguments, or a raising :meth:`Tool.run` are all reported as a
    string so a broken tool can never take down the round.
    """
    by_name = {tool_name(t): t for t in tools or []}
    tool_cls = by_name.get(name)
    if tool_cls is None:
        return f"[tool error: unknown tool {name!r}]"

    if isinstance(arguments, str):
        try:
            data = json.loads(arguments or "{}")
        except ValueError as e:
            return f"[tool error: invalid arguments: {e}]"
    else:
        data = dict(arguments or {})
    data.pop("action", None)

    try:
        instance = tool_cls(**data)
    except ValidationError as e:
        return f"[tool error: {e}]"
    try:
        return str(instance.run())
    except Exception as e:  # noqa: BLE001 — a broken tool must not kill the round
        return f"[tool error: {e}]"
