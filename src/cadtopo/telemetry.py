"""Process-wide accounting for LLM calls: token cost and a verbatim call log.

Two module-level singletons are exported:

  * ``COST``  — a :class:`CostTracker` accumulating per-agent token usage,
    used for cost reporting and per-round/per-task cost deltas.
  * ``CALLS`` — a :class:`CallLog` capturing EVERY LLM call verbatim (full
    request messages + response), used by :mod:`cadtopo.tracing` to build
    complete run traces.

Both are populated at the single choke point where every provider call is
issued (:func:`cadtopo.llm.structured_completion`), so nothing can slip
through uncounted.
"""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class AgentCost:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def copy(self) -> "AgentCost":
        return AgentCost(**vars(self))

    def __sub__(self, other: "AgentCost") -> "AgentCost":
        return AgentCost(
            calls=self.calls - other.calls,
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            completion_tokens=self.completion_tokens - other.completion_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
        )


class CostTracker:
    """Process-wide token accumulator, keyed by agent name AND by model."""

    def __init__(self) -> None:
        self.by_agent: dict[str, AgentCost] = {}
        self.by_model: dict[str, AgentCost] = {}
        self.total: AgentCost = AgentCost()

    def record(self, name: str, usage: Any, model: str | None = None) -> None:
        """Accumulate one LLM call's usage into the per-agent, per-model + global totals.

        ``usage`` is a LiteLLM/OpenAI-style ``response.usage`` object
        (``prompt_tokens``/``completion_tokens``/``total_tokens``). Silently
        ignored when ``None`` or when a provider omits the fields. ``model`` is
        the provider/model string the call ran on (the Backbone's ``model``);
        when given, the same usage is also accumulated under :attr:`by_model`,
        so a run can be broken down — and costed — per model.
        """
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))

        buckets = [self.by_agent.setdefault(name, AgentCost()), self.total]
        if model is not None:
            buckets.append(self.by_model.setdefault(str(model), AgentCost()))
        for bucket in buckets:
            bucket.calls += 1
            bucket.prompt_tokens += prompt
            bucket.completion_tokens += completion
            bucket.total_tokens += total

    def snapshot(self) -> dict[str, AgentCost]:
        """Deep-copy current per-agent state, for a later :meth:`diff_since`."""
        return {name: cost.copy() for name, cost in self.by_agent.items()}

    def diff_since(self, baseline: dict[str, AgentCost]) -> dict[str, AgentCost]:
        """Per-agent cost delta between ``baseline`` and now (unchanged agents omitted)."""
        delta: dict[str, AgentCost] = {}
        for name, cost in self.by_agent.items():
            before = baseline.get(name, AgentCost())
            d = cost - before
            if d.calls > 0:
                delta[name] = d
        return delta

    @staticmethod
    def aggregate(delta: dict[str, AgentCost]) -> AgentCost:
        """Sum a per-agent delta (e.g. from :meth:`diff_since`) into one total."""
        total = AgentCost()
        for ac in delta.values():
            total.calls += ac.calls
            total.prompt_tokens += ac.prompt_tokens
            total.completion_tokens += ac.completion_tokens
            total.total_tokens += ac.total_tokens
        return total


# Module-level singleton — import and use directly, no wiring needed.
COST = CostTracker()


@dataclass
class LLMCall:
    """One completed LLM call, captured verbatim.

    ``messages`` is the exact request sent to the provider and ``response``
    the raw assistant text returned — nothing truncated. Tool-calling turns
    (see :meth:`cadtopo.agent.Agent.step`) additionally carry
    ``tool_calls`` (on the requesting assistant turn) or ``tool_call_id`` (on
    the matching tool-result turn) when present.
    """

    seq: int
    label: str
    model: str
    messages: List[dict]
    response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CallLog:
    """Process-wide, append-only log of every LLM completion.

    Recorded at the single choke point (``structured_completion``), so no
    call — manager review, retry, worker pass — can slip through unrecorded.
    ``label`` tags the current caller (set via :meth:`set_label`); it is
    informational only, the captured messages identify the call regardless.
    """

    def __init__(self) -> None:
        self.calls: List[LLMCall] = []
        self._label: str = "unknown"

    def set_label(self, label: str) -> None:
        """Tag subsequent calls with the current caller (e.g. an agent name)."""
        self._label = label or "unknown"

    def record(self, model: Any, messages: Any, response: str, usage: Any) -> None:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion)) if usage else 0

        clean = []
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            entry = {"role": m.get("role", ""), "content": m.get("content", "")}
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id"):
                entry["tool_call_id"] = m["tool_call_id"]
            clean.append(entry)

        self.calls.append(
            LLMCall(
                seq=len(self.calls) + 1,
                label=self._label,
                model=str(model),
                messages=clean,
                response=response or "",
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
            )
        )

    def mark(self) -> int:
        """Current length — pass to :meth:`since` to slice calls made after it."""
        return len(self.calls)

    def since(self, mark: int) -> List[LLMCall]:
        """Every call recorded since ``mark``, in call order."""
        return self.calls[mark:]


# Module-level singleton — every LLM call in the process is appended here.
CALLS = CallLog()
