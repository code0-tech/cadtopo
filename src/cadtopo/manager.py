"""The manager: DyTopo's meta-agent Π_meta (Sec. 3.5)."""

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from . import llm
from .backbone import Backbone, resolve_backbones
from .logging_utils import get_logger
from .schema import InitialGoal, RoundDecision

_log = get_logger("cadtopo.manager")


class ManagerDecision(BaseModel):
    """The manager's per-round control decision (DyTopo Eq. 13/14).

    Produced by ONE manager call per round (:meth:`Manager.review_round`):
    it reads the global state S_global — the round goal and every agent's
    public message (Eq. 12) — and in a single pass it CHOOSES the round's
    deliverable, scores it, and drafts the next round goal. The manager picks
    ``best_agent`` itself (the agent whose public output is the finished
    artifact answering the request, never an evaluator's verdict/critique);
    ``best_result`` is that agent's output, carried through for the read-out.
    A router coarse score is no longer what fixes the deliverable.

    Halting follows Eq. 14, but with a continuous score instead of a
    self-asserted boolean AND stated in the INVERSE: the model emits
    ``failure_score`` = P(wrong) ∈ [0,1] (its estimated probability the picked
    output is DEFECTIVE), and Φ = ``1 - failure_score`` is derived here. The
    system halts iff ``phi ≥ gamma_success``. Asking for the failure
    probability, not a success score, reframes the judgement as a bug hunt —
    weak backbones inflate self-praise ("looks good, Φ=0.95") and cause
    false-positive halts, but are more discriminating when asked to estimate
    how likely their own artifact is to be broken. ``is_complete`` is the
    *derived* halting bit y(t) — not something the model asserts directly (a
    yes/no bit tends to make weak backbones loop forever, always finding a
    pretext to answer "not complete").
    """

    is_complete: bool
    next_goal: str = ""
    phi: float = 0.0
    success_score: float = 0.0
    best_agent: str = ""
    best_result: str = ""


class CompletedSubTask(BaseModel):
    """A round goal plus the result the topology produced for it."""

    goal: str
    result: str


def aggregate_public(results: Dict[str, str]) -> str:
    """Ordered concatenation of a round's public messages (S_global, Eq. 12).

    Used as the round's deliverable only when :meth:`Manager.review_round`
    couldn't resolve a best agent (unparsable call, or no agent ran).
    """
    return "\n\n".join(f"--- [{name}] ---\n{out}" for name, out in results.items() if out).strip()


def _resolve_best(picked: str, results: Dict[str, str]) -> tuple[str, str]:
    """Resolve the manager's named ``BEST_AGENT`` to ``(name, output)``.

    Matches the picked name exactly, then case-insensitively, then by loose
    containment (so ``"the Developer"`` or ``"Developer agent"`` still lands
    on ``Developer``). Only agents that produced output are eligible. Returns
    ``("", "")`` if nothing usable — the caller then falls back to the first
    agent with output.
    """
    picked = (picked or "").strip().strip("[]").strip()
    if not picked:
        return "", ""
    if results.get(picked):
        return picked, results[picked]
    low = picked.lower()
    for name, out in results.items():
        if out and name.lower() == low:
            return name, out
    for name, out in results.items():
        if out and (name.lower() in low or low in name.lower()):
            return name, out
    return "", ""


def _coerce_str(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return ""
    if not isinstance(value, str):
        return str(value)
    return value.strip()


class Manager:
    """Central manager, run as a closed-loop controller (DyTopo Sec. 3.5).

    Unlike the worker agents, the manager keeps a GLOBAL view: it sits
    outside the routing graph and never receives routed private messages.
    Each round it does exactly ONE thing, execution-free — no code runs, the
    judgement is the backbone's own:

      * :meth:`initial_goal` — ONCE per task, before round 1: turn the task
        into the first round goal C_task(0), so the agents never receive the
        raw request AS the goal.
      * :meth:`review_round` — EVERY round: read the global state and in a
        single pass CHOOSE the round's deliverable (the agent whose output is
        the finished artifact), score it, and draft the next round goal.
        Halting is derived by thresholding Φ against ``gamma_success``.

    There is no separate final-synthesis step and no up-front plan: round
    goals are generated reactively, one at a time, so the system can shift
    from exploration to verification and stop as soon as the request is
    satisfied.

    Like the workers, the manager carries a cost-aware backbone LADDER: a
    :class:`~cadtopo.selection.ModelSelector` picks which rung runs each round's
    control call (see :meth:`cadtopo.selection.ModelSelector.for_control`). The
    manager is the FIRST thing escalated when the team is stuck — a weak
    conductor both misjudges Φ and missteers the next goal.

    :param backbones: The manager's ladder of candidate :class:`Backbone`\\ s
        (cheapest-first). Mutually exclusive with the single-backbone shorthand
        (``api_provider``/``cost_per_token``/``api_key``/``api_url``).
    :param api_provider: Shorthand: LiteLLM model string for a single backbone.
    :param api_key: Shorthand: API key for the single backbone.
    :param cost_per_token: Shorthand: cost of the single backbone.
    :param gamma_success: γ_success (Eq. 14) — acceptance threshold on Φ.
        Halt once the manager's success score reaches it.
    :param control_temperature: Sampling temperature for the control call
        specifically (a judging task, not a generative one). Defaults to
        ``0.0``; pass ``None`` to omit it and use the provider default.
    :param control_extra_params: Arbitrary provider-specific kwargs merged
        into the control call, e.g. ``{"reasoning_effort": "high"}``
        (OpenAI o-series/GPT-5) or ``{"thinking": {...}}`` (Anthropic
        extended thinking) — so review can "think" harder than the workers.
    :param verdict_agent: Name of the evaluator agent (e.g. ``"Tester"``)
        whose public output is surfaced to :meth:`review_round` as a
        dedicated VERDICT block and named the primary evidence for Φ —
        DyTopo's manager-hears-the-tester coupling. Evidence, not a gate:
        the manager still makes the halting decision itself (Π_meta stays
        the only halting authority), it is just told to ground Φ in the
        verdict rather than its own code-reading. ``None`` (default)
        disables the coupling; a round where the verdict agent produced no
        output degrades to the plain review.
    """

    def __init__(
        self,
        api_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: str | None = None,
        backbones: Optional[List[Backbone]] = None,
        cost_per_token: Optional[float] = None,
        gamma_success: float = 0.8,
        control_temperature: float | None = 0.0,
        control_extra_params: Dict[str, Any] | None = None,
        verdict_agent: str | None = None,
    ):
        self.backbones = resolve_backbones("Manager", backbones, api_provider, cost_per_token, api_key, api_url)
        # The rung the selector currently points the control call at (cheapest
        # by default; escalated first when the team is stuck).
        self.active_backbone = self.backbones[0]
        self.gamma_success = gamma_success
        self.control_temperature = control_temperature
        self.control_extra_params = dict(control_extra_params or {})
        self.verdict_agent = verdict_agent

    # Backbone-derived accessors: every control call speaks through whatever
    # rung the selector has currently pointed ``active_backbone`` at.
    @property
    def cost(self) -> float:
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

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def _structured(
        self,
        system: str,
        user: str,
        response_model: Type[BaseModel],
        phase: str,
        extra_params: Dict[str, Any] | None = None,
    ) -> BaseModel:
        """Run one manager completion returning a validated ``response_model``.

        Every manager call is structured JSON, parsed and validated by
        Instructor (see :func:`cadtopo.llm.structured_completion`) — the same
        family every CADTopo call now uses. ``phase`` tags the call for the
        trace recorder (see :meth:`_record`).
        """
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        obj = llm.structured_completion(
            response_model=response_model, messages=messages, model=self.api_provider,
            api_key=self.api_key, api_base=self.api_url, label="Manager", **(extra_params or {}),
        )
        self._record(phase, system, user, obj)
        return obj

    def _record(self, phase: str, system: str, user: str, obj: BaseModel) -> None:  # overridden by RecordingManager
        """Trace hook — a no-op in the base manager."""

    # ------------------------------------------------------------------ #
    # Call 0 (once per task, before round 1): formulate the first round goal
    # ------------------------------------------------------------------ #
    def initial_goal(self, task: str) -> str:
        """Formulate the FIRST round goal C_task(0) from the task (Eq. 13).

        The agents also see the full task separately every round, so this
        goal only STEERS round 1 — it must not restate the task. Fails open
        with a generic first directive so round 1 always has a goal.
        """
        system = (
            "You are Manager, the Workflow Orchestrator. Before any work "
            "begins, set the FIRST round goal for a team of agents who will "
            "solve the TASK below over several rounds. The agents ALSO see the "
            "full task separately every round, so this goal only STEERS the "
            "first round — it says what to focus on now; it does NOT restate "
            "the task.\n"
            "RULES:\n"
            "1. One short imperative command, starting with an action verb. "
            "Never a question.\n"
            "2. Steer, do NOT re-specify: the agents already have the task, so "
            "do not copy its details, examples, or signature into the goal.\n"
            "3. Aim for a first COMPLETE attempt at the task, not a partial "
            "exploration, unless the task is obviously multi-stage."
        )
        user = f"The task the team must solve:\n{task}\n\nState the first round goal now."

        try:
            obj = self._structured(system, user, InitialGoal, phase="initial_goal")
        except Exception as e:  # noqa: BLE001
            _log.error(f"initial_goal failed: {llm.format_llm_error(e)}")
            return "Produce a first complete solution that satisfies the task above."
        goal = _coerce_str(obj.goal)
        return goal or "Produce a first complete solution that satisfies the task above."

    # ------------------------------------------------------------------ #
    # Per-round control: the DyTopo meta-policy Π_meta in ONE pass
    # ------------------------------------------------------------------ #
    def review_round(
        self,
        user_prompt: str,
        round_goal: str,
        results: Dict[str, str],
        completed: List[CompletedSubTask],
        round_num: int,
        max_rounds: int,
    ) -> ManagerDecision:
        """One DyTopo meta-policy pass Π_meta (Eq. 13/14).

        The deliverable IS chosen here: the manager reads the global state —
        the round goal, every agent's public message (σ(t)-ordered), and the
        original request — and in a SINGLE execution-free call names the
        agent whose output is the round's deliverable (``BEST_AGENT``), scores
        that deliverable, and drafts the next round goal. The manager is told
        to pick the finished artifact that answers the request, never an
        evaluator's verdict, a review, or a test report ABOUT it — those are
        evidence, not the deliverable. If the manager names no usable agent,
        the deliverable falls back to the first agent that produced output
        this round (σ(t) order).

        ``completed``/``round_num``/``max_rounds`` are accepted for signature
        stability (the orchestrator has them handy); they are not fed to the
        model — the decision is scoped to the global state alone.

        If ``self.verdict_agent`` produced output this round, that output is
        additionally surfaced as a dedicated VERDICT block and named the
        primary evidence for Φ (see ``__init__``).

        Fail-safe: on an unparsable call, or a continue without a usable
        ``next_goal``, the loop HALTS so a deliverable is always produced.
        """
        verdict = (results.get(self.verdict_agent) or "").strip() if self.verdict_agent else ""

        # Fail-open deliverable: if the manager names no usable agent, fall
        # back to the first agent that produced output this round (σ(t) order).
        fallback_agent, fallback_result = "", ""
        for name, out in results.items():
            if out:
                fallback_agent, fallback_result = name, out
                break

        verdict_rule = ""
        if verdict:
            verdict_rule = (
                "\n5. A TESTER VERDICT section is present: it comes from the "
                "dedicated evaluator agent, which checks the candidate against "
                "the request's own stated examples (executing it where it "
                "can). Treat it as the PRIMARY evidence for failure_score — do "
                "not overrule it with your own reading of the code. A verdict "
                "of INCORRECT naming a concrete failing case is a DEMONSTRATED "
                "defect: set failure_score HIGH (near 1) and target NEXT_GOAL "
                "at exactly that failing case. Only discount the verdict if it "
                "clearly refers to an older or different candidate than the "
                "DELIVERABLE shown above."
            )
        system = (
            "You are Manager, the Workflow Orchestrator (the meta-agent). You "
            "read the WHOLE round — this round's goal and every agent's public "
            "output. You do TWO things: (1) CHOOSE which agent's output is this "
            "round's DELIVERABLE (best_agent), and (2) actively HUNT for a "
            "defect in that DELIVERABLE and rate it from BOTH directions — "
            "failure_score (how likely it is WRONG) and success_score (how "
            "likely it is RIGHT) — plus what the next round should fix "
            "(next_goal).\n"
            "RULES:\n"
            "1. best_agent: pick the ONE agent whose public output IS the "
            "finished artifact that answers the ORIGINAL request (e.g. the "
            "actual solution or code), naming it verbatim from the list below. "
            "An evaluator's verdict, a reviewer's critique, or a test report is "
            "EVIDENCE about the deliverable, never the deliverable itself — do "
            "NOT pick such an agent, even when it is the most informative "
            "output.\n"
            "2. failure_score = P(wrong): your job here is NOT to confirm the "
            "team did well — it is to FIND THE BUG. Trace the DELIVERABLE "
            "against the request's own signature, description and stated "
            "examples, actively looking for an input where it returns the wrong "
            "result. Most first-attempt outputs contain at least one bug, so "
            "start from the assumption it is broken and try to prove it. Report "
            "P(wrong) ∈ [0,1]: keep it HIGH unless you traced EVERY stated "
            "example and case and found NO input that breaks it — only then may "
            "failure_score approach 0. A vague 'looks fine' is not a passing "
            "trace. Then ALSO give success_score = P(right) ∈ [0,1] — the same "
            "verdict from the opposite side. Judge it on its own merits from the "
            "same trace; do NOT just fill in 1 - failure_score. If the two "
            "disagree, that itself signals you are not sure — let them fall where "
            "your trace honestly puts them.\n"
            "3. next_goal steers the next round on the SAME task. The agents "
            "already have the full task, so name the concrete fix — do NOT "
            "restate the task, and never phrase it as a question. Leave a brief "
            "directive even if you think it is done.\n"
            "4. 'Could be faster', 'could be more robust', input validation, or "
            "edge cases the request does NOT mention are NOT defects — they "
            "must NOT raise failure_score. Only a wrong result on a case the "
            "request cares about counts."
            f"{verdict_rule}"
        )

        public_block = (
            "\n\n".join(f"[{name}]:\n{out or '<no output>'}" for name, out in results.items())
            if results else "<no agent produced output this round>"
        )
        candidate_names = ", ".join(name for name, out in results.items() if out) or "<none>"
        verdict_block = (
            f"TESTER VERDICT (from {self.verdict_agent}, the evaluator — primary evidence for the score):\n"
            f"{verdict}\n\n"
            if verdict else ""
        )
        user = (
            f"The original user request:\n{user_prompt}\n\n"
            f"This round's goal (what the agents were told to focus on this round):\n"
            f"{round_goal or '<none — round 1: agents saw only the task>'}\n\n"
            f"This round's agent outputs:\n{public_block}\n\n"
            f"{verdict_block}"
            f"Choose the DELIVERABLE by naming exactly one of these agents: {candidate_names}. "
            f"Then, in bug_hunt, actually TRACE that deliverable on the request's stated examples "
            f"BEFORE you score — compute its output on the hardest case and compare to the expected one. "
            f"Only then report BOTH scores as the CONSEQUENCE of that trace: failure_score = "
            f"P(wrong) ∈ [0,1] (high unless every stated case matched) AND success_score = "
            f"P(right) ∈ [0,1], judged independently from the opposite direction. Finally state "
            f"the next round goal."
        )

        extra_params: Dict[str, Any] = dict(self.control_extra_params)
        if self.control_temperature is not None:
            extra_params.setdefault("temperature", self.control_temperature)

        try:
            decision = self._structured(system, user, RoundDecision, phase="control", extra_params=extra_params)
        except Exception as e:  # noqa: BLE001
            _log.error(f"review_round failed: {llm.format_llm_error(e)} — halting loop.")
            return ManagerDecision(is_complete=True, phi=1.0, success_score=1.0, best_agent=fallback_agent, best_result=fallback_result)

        # Resolve the manager's chosen deliverable; fail open to the fallback.
        best_agent, best_result = _resolve_best(decision.best_agent, results)
        if not best_result:
            best_agent, best_result = fallback_agent, fallback_result

        # Φ averages the two framings the model reports in one call — the
        # complement of P(wrong) and the direct P(right) — so one-sided
        # over/under-confidence cancels (self-consistency, see RoundDecision).
        phi = ((1.0 - decision.failure_score) + decision.success_score) / 2.0

        # Eq. 14: y(t) = 1 iff Φ ≥ γ_success.
        is_complete = phi >= self.gamma_success
        if is_complete:
            _log.info(
                f"Manager: task COMPLETE after round {round_num} "
                f"(Φ={phi:.2f} ≥ γ={self.gamma_success:.2f}, best={best_agent!r})."
            )
            return ManagerDecision(is_complete=True, phi=phi, success_score=phi, best_agent=best_agent, best_result=best_result)

        next_goal = _coerce_str(decision.next_goal)
        if not next_goal:
            _log.warning(f"review_round: Φ={phi:.2f} < γ={self.gamma_success:.2f} but no usable next_goal — halting loop.")
            return ManagerDecision(is_complete=True, phi=phi, success_score=phi, best_agent=best_agent, best_result=best_result)

        _log.info(f"Manager: continue (Φ={phi:.2f} < γ={self.gamma_success:.2f}, best={best_agent!r}) — next goal: {next_goal!r}")
        return ManagerDecision(is_complete=False, next_goal=next_goal, phi=phi, success_score=phi, best_agent=best_agent, best_result=best_result)
