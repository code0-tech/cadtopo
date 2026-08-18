"""Single choke point for every LLM completion call.

Every call site in CADTopo (worker passes, manager reviews) goes through
:func:`structured_completion` instead of calling ``litellm.completion``
directly, so all of them get the same retry policy, land in the global
:data:`cadtopo.telemetry.CALLS` log, and return a validated Pydantic model.

Structured output is provided by Instructor wrapping LiteLLM. The wire mode is
chosen by the ``CADTOPO_INSTRUCTOR_MODE`` env var (default ``JSON``): ``JSON``
renders the response model's JSON Schema into the prompt and asks for a JSON
object back — which works for every provider, including those that cannot be
handed a schema natively — while modes like ``TOOLS`` / ``JSON_SCHEMA`` use the
provider's native structured-output feature where it is reliable. Either way the
model only ever emits schema-valid JSON; there is no bespoke delimiter parser.
"""

import logging
import os
from typing import Any, Type, TypeVar

import instructor
import litellm
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .telemetry import CALLS, COST

litellm.suppress_debug_info = True
# Instructor logs the full "Max retries exceeded … <failed_attempts>" dump (every
# raw completion) at ERROR on a validation give-up. We surface a concise summary
# ourselves (see format_llm_error), so silence its verbose channel.
logging.getLogger("instructor").setLevel(logging.CRITICAL)
# LiteLLM warns per call on providers that return finish_reason='error' (e.g. a
# reasoning model whose content came back empty) — one line per retry, pure
# noise. The condition still surfaces as an empty-content parse failure we handle.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)


def format_llm_error(exc: BaseException) -> str:
    """A short, log-safe one-liner for an LLM failure.

    Instructor's :class:`InstructorRetryException` stringifies to a giant
    ``<failed_attempts>`` block containing every raw ``ModelResponse`` — useless
    noise in a log. This unwraps it: it reports the retry count and the ROOT
    error (the actual validation or transport failure at the end of the
    ``__cause__`` chain), collapsing a Pydantic ``ValidationError`` to its
    ``loc: msg`` essentials.
    """
    attempts = getattr(exc, "n_attempts", None)
    root = exc
    while getattr(root, "__cause__", None) is not None:
        root = root.__cause__  # type: ignore[assignment]

    if isinstance(root, ValidationError):
        errs = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in root.errors()
        )
        detail = f"response did not match schema ({errs})"
    else:
        detail = f"{type(root).__name__}: {root}"

    prefix = f"gave up after {attempts} attempts — " if attempts else ""
    msg = prefix + detail
    return msg if len(msg) <= 300 else msg[:297] + "..."

# Exceptions worth retrying. RateLimitError dominates on free OpenRouter/Groq
# tiers; the rest cover transient provider/transport hiccups. (Instructor's own
# ``max_retries`` separately handles schema-validation re-asks — those are NOT
# retried here.)
_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.ServiceUnavailableError,
)

# Providers that enforce JSON server-side (Groq, some OpenRouter upstreams)
# occasionally fail to emit valid JSON for a given sampling and surface it as a
# bare ``APIError`` ("Failed to generate JSON") — NOT a transport error, and NOT
# a Pydantic ``ValidationError`` either, so neither our transport-retry gate nor
# Instructor's ``max_retries`` re-ask would catch it and the call would die after
# one attempt. A fresh sampling almost always succeeds, so we treat these
# message-identified provider hiccups as transient too.
_TRANSIENT_APIERROR_MARKERS: tuple[str, ...] = (
    "failed to generate json",
)


def _is_transient(exc: BaseException) -> bool:
    """True if ``exc`` is (or wraps) a transient transport/provider error.

    Instructor wraps the underlying provider error in an
    ``InstructorRetryException``, so the error we care about is found by walking
    the ``__cause__`` chain rather than matching the top type. Besides the
    transport exception types, a re-samplable server-side JSON-generation failure
    (a plain ``APIError`` whose text matches ``_TRANSIENT_APIERROR_MARKERS``) also
    counts as transient.
    """
    seen = exc
    while seen is not None:
        if isinstance(seen, _RETRY_EXCEPTIONS):
            return True
        if isinstance(seen, litellm.APIError):
            text = str(seen).lower()
            if any(marker in text for marker in _TRANSIENT_APIERROR_MARKERS):
                return True
        seen = seen.__cause__
    return False


T = TypeVar("T", bound=BaseModel)


def _instructor_mode() -> "instructor.Mode":
    name = os.environ.get("CADTOPO_INSTRUCTOR_MODE", "JSON").upper()
    return getattr(instructor.Mode, name, instructor.Mode.JSON)


# The client is built from a lambda that resolves ``litellm.completion`` at call
# time, so tests can monkeypatch ``cadtopo.llm.litellm.completion`` and have the
# patch take effect through Instructor.
_client = instructor.from_litellm(lambda **kw: litellm.completion(**kw), mode=_instructor_mode())


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
def structured_completion(
    *,
    response_model: Type[T],
    messages: list[dict[str, Any]],
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    label: str | None = None,
    max_retries: int = 2,
    **extra: Any,
) -> T:
    """Call the provider (via Instructor) and return a validated ``response_model``.

    Transport failures retry with exponential backoff (6 attempts, 2s→30s);
    schema-validation failures are re-asked by Instructor up to ``max_retries``.
    Token usage and a verbatim request/response record are captured at this one
    choke point, so nothing slips by uncounted. ``extra`` is forwarded to
    ``litellm.completion`` (e.g. ``temperature``, ``reasoning_effort``).
    """
    if label:
        CALLS.set_label(label)
    obj, raw = _client.chat.completions.create_with_completion(
        model=model,
        messages=messages,
        response_model=response_model,
        api_key=api_key,
        api_base=api_base,
        max_retries=max_retries,
        drop_params=True,  # a provider that doesn't understand a knob just ignores it
        **extra,
    )
    usage = getattr(raw, "usage", None)
    COST.record(label or "unknown", usage, model=model)
    try:
        content = raw.choices[0].message.content or ""
    except (AttributeError, IndexError):
        content = ""
    if not content:  # native-tool modes may leave content empty — log the parsed object
        try:
            content = obj.model_dump_json()
        except Exception:  # noqa: BLE001
            content = str(obj)
    CALLS.record(model=model, messages=messages, response=content, usage=usage)
    return obj


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
def text_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    label: str | None = None,
    **extra: Any,
) -> str:
    """One plain-text completion: return the assistant's raw text, no schema.

    Used for an agent's WORK deliverable, which must NOT be marshalled through a
    JSON string — doing so makes weak models mangle the escaping of quotes and
    newlines and corrupts code (broken docstrings, truncated bodies). Here the
    whole response IS the deliverable. Telemetry is recorded at this same choke
    point and transport failures retry with the shared backoff policy.
    """
    if label:
        CALLS.set_label(label)
    raw = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        api_base=api_base,
        drop_params=True,  # a provider that doesn't understand a knob just ignores it
        **extra,
    )
    usage = getattr(raw, "usage", None)
    COST.record(label or "unknown", usage, model=model)
    try:
        content = raw.choices[0].message.content or ""
    except (AttributeError, IndexError):
        content = ""
    CALLS.record(model=model, messages=messages, response=content, usage=usage)
    return content


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
def tool_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    api_key: str | None = None,
    api_base: str | None = None,
    label: str | None = None,
    tool_choice: str = "auto",
    **extra: Any,
) -> Any:
    """One native tool-calling turn: return the provider's raw response message.

    Unlike :func:`structured_completion` this does NOT coerce the reply into a
    Pydantic model — the tools are offered through the provider's standard
    ``tools=[...]`` interface, so the model answers with either ``tool_calls``
    (executed harness-side; see :func:`cadtopo.tools.dispatch`) or plain content
    signalling it is done. The returned object is ``choices[0].message`` with its
    ``content`` and ``tool_calls`` attributes. Telemetry is recorded at this same
    choke point; transport failures retry with the shared backoff policy.
    """
    if label:
        CALLS.set_label(label)
    raw = litellm.completion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        api_key=api_key,
        api_base=api_base,
        drop_params=True,  # a provider that doesn't understand a knob just ignores it
        **extra,
    )
    usage = getattr(raw, "usage", None)
    COST.record(label or "unknown", usage, model=model)
    message = raw.choices[0].message
    content = getattr(message, "content", None) or ""
    if not content:  # a tool-call turn carries no prose — log the calls it requested
        calls = getattr(message, "tool_calls", None) or []
        content = "; ".join(
            f"{getattr(tc.function, 'name', '?')}({getattr(tc.function, 'arguments', '')})"
            for tc in calls
        ) or "[no content]"
    CALLS.record(model=model, messages=messages, response=content, usage=usage)
    return message
