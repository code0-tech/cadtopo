"""The worker agent: a single specialised sub-model in the CADTopo pool."""

from typing import Any, List, Optional, Type

from cadtopo.schema import AgentStep, Descriptors, Tool
from . import llm
from .backbone import Backbone, resolve_backbones
from .logging_utils import get_logger
from .telemetry import CALLS
from .tools import dispatch, tool_schemas

_log = get_logger("cadtopo.agent")


class Agent:
    """A specialised agent with its own backbone, role, and (optionally) tools.

    :param name: Unique name of the agent.
    :param skill_definition: Static description of the agent's capabilities
        (S_i), used by the router's Stage 1 coarse selection.
    :param backbones: The agent's ladder of candidate :class:`Backbone`\\ s
        (each a model + its cost). Stored sorted cheapest-first; a
        :class:`~cadtopo.selection.ModelSelector` sets :attr:`active_backbone`
        per round. Mutually exclusive with the single-backbone shorthand
        (``api_provider``/``cost_per_token``/``api_key``/``api_url``), which is
        kept for backward compatibility and wraps one :class:`Backbone`.
    :param cost_per_token: Shorthand: inference cost of the single backbone
        (Cost_i). Ignored when ``backbones`` is given.
    :param api_provider: Shorthand: LiteLLM model/provider string for the single
        backbone. Ignored when ``backbones`` is given.
    :param api_url: Shorthand: optional custom base URL for the single backbone.
    :param api_key: Shorthand: API key for the single backbone.
    :param system_prompt: DyTopo-style strict role prompt sent as the ``system``
        message on every call. ``None`` runs without a role message.
    :param tools: A list of :class:`cadtopo.schema.Tool` subclasses the agent may
        call. They are offered to the provider through the STANDARD native
        tool-calling interface (``tools=[...]``; see
        :func:`cadtopo.tools.tool_schemas`), so the model selects and fills a
        tool through the provider's own machinery. Tools are executed
        harness-side between turns. ``None`` (default) disables tool-calling —
        the pass is a single plain-text completion.
    :param max_tool_iterations: Hard cap on native tool-calling round-trips
        within one pass. When exhausted (or the model stops calling tools) the
        pass ends with the model's plain-text deliverable.
    :param mandatory: If True the agent bypasses the router's Stage 1 θ gate and
        is active EVERY round (a CADTopo extension).
    """

    def __init__(
            self,
            name: str,
            skill_definition: str,
            backbones: Optional[List[Backbone]] = None,
            cost_per_token: Optional[float] = None,
            api_provider: Optional[str] = None,
            api_url: Optional[str] = None,
            api_key: Optional[str] = None,
            system_prompt: Optional[str] = None,
            tools: Optional[List[Type[Tool]]] = None,
            max_tool_iterations: int = 4,
            mandatory: bool = False,
    ):
        self.name = name
        self.skill_definition = skill_definition
        self.backbones = resolve_backbones(name, backbones, api_provider, cost_per_token, api_key, api_url)
        # The rung currently selected for this agent. Defaults to the cheapest;
        # a ModelSelector re-points it each round (see cadtopo.selection).
        self.active_backbone = self.backbones[0]
        self.system_prompt = system_prompt
        self.tools = list(tools or [])
        self.max_tool_iterations = max_tool_iterations
        self.mandatory = mandatory

    # Backbone-derived accessors: the agent always speaks through whatever rung
    # a ModelSelector has currently pointed ``active_backbone`` at.
    @property
    def cost(self) -> float:
        """Cost (per 1k tokens) of the CURRENTLY selected backbone (Cost_i)."""
        return self.active_backbone.cost

    @property
    def api_provider(self) -> str:
        return self.active_backbone.model

    @property
    def api_key(self) -> Optional[str]:
        return self.active_backbone.api_key

    @property
    def api_url(self) -> Optional[str]:
        return self.active_backbone.api_url

    # Phase-1 descriptor instruction (Sec. 3.3): a lightweight call returning
    # ONLY the routing descriptors, with no work and no tools.
    _DESCRIBE_GUIDANCE = (
        "Before doing any work this round, declare ONLY your routing descriptors "
        "so the coordinator can wire up who hands off to whom. Do NOT solve the "
        "task and write no code in this response. QUERY and KEY are plain-English "
        "sentences describing your need and your offer. FAILURE_RISK is your "
        "honest P(wrong) for THIS round — the probability you canNOT contribute "
        "correctly; hunt for what could go wrong rather than assuming success."
    )

    # Phase-2 work instruction (Sec. 3.2.1): respond with the deliverable itself,
    # as plain text — no JSON envelope (that corrupts code escaping on weak
    # backbones), no private note; the whole response is the public deliverable.
    _STEP_GUIDANCE = (
        "Respond with your contribution to the round goal directly: the finished "
        "deliverable itself, in the shape your role requires (e.g. the complete "
        "code). Write nothing else — no preamble, no commentary."
    )

    _TOOL_GUIDANCE = (
        "You have tools available through the tool-calling interface. Call a tool "
        "whenever it helps you get the round done; you may call several across "
        "turns. Once you have what you need, stop calling tools and reply with "
        "your final deliverable."
    )

    def describe(
            self,
            task: str,
            round_goal: str,
            history: Optional[list[dict[str, str]]] = None,
    ) -> AgentStep:
        """Phase 1 of a round: declare routing descriptors only (Sec. 3.3).

        Returns the agent's QUERY/KEY plus a confidence for THIS round; any work
        text is irrelevant (the response model has no such field). The confidence
        is DERIVED from the inverted self-report: the model states FAILURE_RISK =
        P(wrong) and confidence = ``1 - failure_risk`` (the same bug-hunt
        inversion the manager uses for Φ — a positively framed self-accuracy is
        systematically inflated). Fails open with neutral descriptors so routing
        still proceeds if the call fails.
        """
        history = history or []
        role = self.system_prompt or ""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "\n\n".join(p for p in (role, self._DESCRIBE_GUIDANCE) if p)},
            {"role": "user", "content": f"Task (constant across all rounds):\n\n{task}"},
        ]
        messages.extend(history)
        if round_goal:
            messages.append(
                {"role": "user", "content": f"This round's goal (focus for this round only):\n\n{round_goal}"})

        try:
            obj = llm.structured_completion(
                response_model=Descriptors, messages=messages, model=self.api_provider,
                api_key=self.api_key, api_base=self.api_url, label=self.name,
            )
            return AgentStep(query=obj.query, key=obj.key, accuracy=1.0 - obj.failure_risk)
        except Exception as e:  # noqa: BLE001 — keep the round alive
            _log.error(f"{self.name}: describe failed: {llm.format_llm_error(e)}")
            return AgentStep(query="", key="", accuracy=0.5)

    def step(
            self,
            task: str,
            round_goal: str,
            history: Optional[list[dict[str, str]]] = None,
    ) -> AgentStep:
        """Run one DyTopo round pass (Eq. 2), conditioned on the local state (Eq. 1).

        The local state maps to separate messages: role + guidance → ``system``;
        the constant task → its own ``user`` turn before the history; memory H_i
        → ``history``; the round goal → the final ``user`` turn (omitted in round
        1). With tools set, this one pass may issue several native tool-calling
        round-trips (capped at ``max_tool_iterations``) before the final answer;
        from the round protocol's view it is still a single :class:`AgentStep`.
        """
        history = history or []
        role = self.system_prompt or ""
        guidance = [self._STEP_GUIDANCE]
        if self.tools:
            guidance.append(self._TOOL_GUIDANCE)
        system = "\n\n".join(p for p in ([role] + guidance) if p)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task (constant across all rounds):\n\n{task}"},
        ]
        messages.extend(history)
        if round_goal:
            messages.append(
                {"role": "user", "content": f"This round's goal (focus for this round only):\n\n{round_goal}"})

        try:
            return self._run_pass(messages)
        except Exception as e:  # noqa: BLE001 — keep the round alive
            _log.error(f"{self.name}: step failed: {llm.format_llm_error(e)}")
            return AgentStep(public=f"[ERROR: agent {self.name} could not contribute.]", accuracy=0.0)

    # ------------------------------------------------------------------ #
    # Tool-calling loop
    # ------------------------------------------------------------------ #
    def _text(self, messages: list[dict[str, Any]]) -> str:
        return llm.text_completion(
            messages=messages, model=self.api_provider,
            api_key=self.api_key, api_base=self.api_url, label=self.name,
        )

    def _run_pass(self, messages: list[dict[str, Any]]) -> AgentStep:
        """Drive native tool calls, then return the model's plain-text deliverable.

        With no tools the pass is a single plain-text completion — the whole
        reply IS the deliverable (no schema). With tools, each iteration is one
        native tool-calling turn: if the model returns ``tool_calls`` they are
        executed harness-side (de-duped per pass by (name, arguments), logged,
        and fed back as ``tool`` messages) and the loop continues; if it returns
        no tool calls, that turn's own content is the deliverable. If the loop is
        exhausted still calling tools, one final plain-text completion produces
        the deliverable.
        """
        CALLS.set_label(self.name)
        if not self.tools:
            return AgentStep(public=self._text(messages))

        schemas = tool_schemas(self.tools)
        tool_log: List[dict[str, str]] = []
        seen: dict[tuple, str] = {}  # same-pass de-dupe of identical (name, args) calls
        final_text = ""

        for _ in range(self.max_tool_iterations):
            message = llm.tool_completion(
                messages=messages, model=self.api_provider, tools=schemas,
                api_key=self.api_key, api_base=self.api_url, label=self.name,
            )
            calls = getattr(message, "tool_calls", None)
            if not calls:
                # Model stopped calling tools — its reply IS the deliverable.
                final_text = getattr(message, "content", None) or ""
                break

            messages.append(self._assistant_tool_turn(message, calls))
            for tc in calls:
                name = tc.function.name
                args_json = tc.function.arguments or "{}"
                dedupe_key = (name, args_json)
                if dedupe_key in seen:
                    result = seen[dedupe_key]
                    _log.info(f"{self.name}: tool call {name}({args_json}) -> (cached) {result[:200]}")
                else:
                    result = dispatch(self.tools, name, args_json)
                    seen[dedupe_key] = result
                    _log.info(f"{self.name}: tool call {name}({args_json}) -> {result[:200]}")
                tool_log.append({"name": name, "arguments": args_json, "result": result[:500]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tc, "id", None) or name,
                    "content": result,
                })

        if not final_text:  # loop exhausted (or empty final turn) — ask once for the deliverable
            messages.append({"role": "user", "content": "Do NOT call any more tools. Give your final deliverable now."})
            final_text = self._text(messages)
        return AgentStep(public=final_text, tool_calls=tool_log)

    @staticmethod
    def _assistant_tool_turn(message: Any, calls: Any) -> dict[str, Any]:
        """Re-serialise the model's tool-call turn as an OpenAI ``assistant`` message.

        Keeping the assistant turn (with its ``tool_calls``) in the transcript is
        required for the follow-up ``tool`` result messages to be valid.
        """
        return {
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [
                {
                    "id": getattr(tc, "id", None) or tc.function.name,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in calls
            ],
        }
