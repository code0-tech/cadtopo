from typing import Type, Union

from pydantic import BaseModel


class Tool(BaseModel):
    """Base class for an executable tool.

    Subclasses declare their arguments as fields and a discriminator via
    ``action: Literal["<tool_name>"]``, and implement :meth:`run`. Tools are
    handed to the provider through the standard native tool-calling interface
    (``tools=[...]``; see :func:`cadtopo.tools.tool_schemas`), so the model
    selects and fills a tool through the provider's own machinery. Tools are
    executed harness-side between turns; the agent's deliverable is the model's
    plain-text reply once it stops calling tools (there is no response schema).
    """

    def run(self) -> str:  # pragma: no cover - overridden by concrete tools
        raise NotImplementedError


def tool_name(tool: Union[Type[Tool], Tool]) -> str:
    """The tool's wire name: the default of its ``action`` discriminator field.

    Accepts either the ``Tool`` subclass or an instance. Falls back to the
    class name if a subclass forgot to declare an ``action`` default.
    """
    cls = tool if isinstance(tool, type) else type(tool)
    field = cls.model_fields.get("action")
    default = getattr(field, "default", None) if field is not None else None
    return default if isinstance(default, str) and default else cls.__name__
