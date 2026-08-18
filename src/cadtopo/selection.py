"""Cost-aware backbone selection across the whole system, per round.

Both the workers and the manager carry a LADDER of
:class:`~cadtopo.backbone.Backbone`\\ s (cheap → expensive). Each round, every
component picks its own rung — the manager and each agent SELF-MANAGE, on their
own signal, with no priority ordering between them.

The default :class:`CostAwareSelector` turns "which rung?" into a single
cost-aware, budget-aware decision instead of a fixed confidence threshold:

  * **Signal → risk.** An agent's risk is ``1 - confidence`` (its inverted
    Phase-1 self-report); the manager's risk is how far the previous round's Φ
    fell below γ_success. Higher risk ⇒ aim higher up the ladder.
  * **Log-cost axis.** Rungs are placed on a LOG-cost axis, not an index axis,
    so a jump 0.1→1 is a smaller move than 1→50 — exponential prices are
    respected, and the "threshold" is *"which price jump does my risk justify?"*
    rather than a flat number.
  * **Budget → urgency.** Fewer rounds left ⇒ higher urgency ⇒ the same risk
    reaches a higher rung *now* (no time to test a cheaper one first); many
    rounds left ⇒ climb gradually and re-decide next round.

The chosen rung is recomputed from the CURRENT signal every round, so it needs
no persistent escalation state and de-escalates automatically when the signal
recovers. A low-confidence agent is therefore upgraded the SAME round its own
confidence drops — nothing waits on the manager.

A trained RouteLLM-style router (matrix factorisation / BERT classifier) can be
dropped in later as an alternative :class:`ModelSelector` behind the same
interface, without touching the agents, the manager, or the orchestrator.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Protocol, TYPE_CHECKING, runtime_checkable

from .backbone import Backbone
from .logging_utils import get_logger

if TYPE_CHECKING:
    from .agent import Agent
    from .manager import Manager

_log = get_logger("cadtopo.selection")

# Urgency is a TIME GAIN on the risk, ramping urgency_min -> urgency_max over the
# rounds. Early (urgency_min < 1) it damps: willingness < risk, so we test cheap
# first. Late (urgency_max > 1) it AMPLIFIES: willingness > risk, so a lukewarm
# but unresolved signal (e.g. Φ=0.5) is pushed over the escalation threshold as
# the budget runs out. The ramp is convex (frac**URGENCY_EXP) so the >1 gain
# stays concentrated in the final rounds and early behaviour is unchanged.
# Multiplicative on purpose: risk 0 (confident / Φ≥γ) stays 0 — never force-
# escalated, only *residual* risk is amplified by time pressure. All tunable.
URGENCY_MIN = 0.5
URGENCY_MAX = 2.0
URGENCY_EXP = 2.0


@runtime_checkable
class ModelSelector(Protocol):
    """Chooses which backbone the manager and each agent run on, per round.

    ``bind`` wires the selector to the live pool; ``reset`` clears the tiny bit
    of cross-round state at the start of a task. ``for_describe``/``for_work``
    point an agent at its Phase-1/Phase-2 backbone, ``for_control`` points the
    manager at its control-call backbone, and ``observe`` records the round's Φ
    (the manager's next-round risk signal).
    """

    def bind(self, manager: "Manager", agents: List["Agent"]) -> None: ...

    def reset(self) -> None: ...

    def for_describe(self, agent: "Agent") -> Backbone: ...

    def for_work(self, agent: "Agent", confidence: Optional[float] = None,
                 round_num: int = 1, max_rounds: int = 1) -> Backbone: ...

    def for_control(self, manager: "Manager", round_num: int = 1, max_rounds: int = 1) -> Backbone: ...

    def observe(self, phi: float) -> None: ...


class CostAwareSelector:
    """Training-free, per-component cost-aware selector (the default).

    Every round each component's rung is recomputed directly from its current
    risk signal, the ladder's log-costs, and the remaining round budget — see
    the module docstring. The only state is the last Φ, used as the manager's
    risk signal next round.

    :param urgency_min: Urgency with all rounds still ahead (<1 ⇒ damps, test
        cheap first).
    :param urgency_max: Urgency on the final round (>1 ⇒ amplifies, so a
        persistent lukewarm signal is forced up a rung before the budget ends).
    :param urgency_exp: Convexity of the ramp; >1 keeps the >1 gain in the late
        rounds so early behaviour is unchanged.
    """

    def __init__(self, urgency_min: float = URGENCY_MIN,
                 urgency_max: float = URGENCY_MAX, urgency_exp: float = URGENCY_EXP):
        self.urgency_min = urgency_min
        self.urgency_max = urgency_max
        self.urgency_exp = urgency_exp
        self._last_phi: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Wiring / lifecycle
    # ------------------------------------------------------------------ #
    def bind(self, manager: "Manager", agents: List["Agent"]) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear cross-round state — called per task so nothing leaks between
        independent problems."""
        self._last_phi = None

    # ------------------------------------------------------------------ #
    # The one decision: risk + budget + log-cost → rung
    # ------------------------------------------------------------------ #
    def _urgency(self, round_num: int, max_rounds: int) -> float:
        """Grows from ``urgency_min`` (all rounds ahead) to ``urgency_max``
        (final round) along a convex ``frac**urgency_exp`` ramp. Passing
        ``urgency_max`` > 1 lets late rounds amplify the risk past its raw value,
        so a stalled lukewarm signal escalates before the budget ends."""
        frac = min(max(round_num / max(max_rounds, 1), 0.0), 1.0)
        return self.urgency_min + (self.urgency_max - self.urgency_min) * (frac ** self.urgency_exp)

    def _target_rung(self, risk: float, backbones: List[Backbone], round_num: int, max_rounds: int) -> int:
        """Rung whose LOG-cost position best matches ``risk * urgency``.

        Rungs sit on a log-cost axis normalised to [0, 1] (cheapest 0, dearest
        1), so exponentially-spaced prices (0.1 / 1 / 50) are respected. The
        willingness ``risk * urgency`` is a position on that axis; we pick the
        nearest rung (ties go UP, toward the stronger model). Low willingness
        lands on a cheap rung, high willingness reaches the dear one.
        """
        n = len(backbones)
        if n == 1:
            return 0
        levels = self._normalized_log_costs(backbones)
        willingness = min(max(risk * self._urgency(round_num, max_rounds), 0.0), 1.0)
        best, best_dist = 0, abs(levels[0] - willingness)
        for k in range(1, n):
            dist = abs(levels[k] - willingness)
            if dist <= best_dist:  # <= : on a tie prefer the stronger rung
                best, best_dist = k, dist
        return best

    @staticmethod
    def _normalized_log_costs(backbones: List[Backbone]) -> List[float]:
        logs = [math.log(max(b.cost, 1e-9)) for b in backbones]
        lo, hi = min(logs), max(logs)
        span = hi - lo
        if span <= 0:
            return [0.0 for _ in backbones]
        return [(lg - lo) / span for lg in logs]

    # ------------------------------------------------------------------ #
    # Per-round selection
    # ------------------------------------------------------------------ #
    def for_describe(self, agent: "Agent") -> Backbone:
        """Phase 1 is a light, work-free call — always the cheapest rung."""
        agent.active_backbone = agent.backbones[0]
        if len(agent.backbones) > 1:
            _log.info(
                f"  {agent.name:<14} {'describe':<8} → {agent.active_backbone.model} "
                f"(rung 0/{len(agent.backbones) - 1}, cost={agent.active_backbone.cost})"
            )
        return agent.active_backbone

    def for_work(self, agent: "Agent", confidence: Optional[float] = None,
                 round_num: int = 1, max_rounds: int = 1) -> Backbone:
        """Pick the Phase-2 work rung from THIS round's confidence (risk =
        ``1 - confidence``), the ladder's log-costs, and the remaining budget.

        Immediate: a low-confidence agent gets a stronger model this very round,
        with no dependence on the manager or other agents.
        """
        risk = 1.0 - (confidence if confidence is not None else 1.0)
        n = len(agent.backbones)
        k = self._target_rung(risk, agent.backbones, round_num, max_rounds)
        agent.active_backbone = agent.backbones[k]
        if n > 1:
            _log.info(
                f"  {agent.name:<14} {'work':<8} → {agent.active_backbone.model} "
                f"(rung {k}/{n - 1}, cost={agent.active_backbone.cost}, conf={1 - risk:.2f})"
            )
        return agent.active_backbone

    def for_control(self, manager: "Manager", round_num: int = 1, max_rounds: int = 1) -> Backbone:
        """Pick the manager's control rung from the PREVIOUS round's Φ: risk is
        how far it fell below γ_success. Round 1 (no Φ yet) → cheapest."""
        gamma = getattr(manager, "gamma_success", 0.8) or 0.8
        risk = 0.0 if self._last_phi is None else min(max((gamma - self._last_phi) / gamma, 0.0), 1.0)
        n = len(manager.backbones)
        k = self._target_rung(risk, manager.backbones, round_num, max_rounds)
        manager.active_backbone = manager.backbones[k]
        if n > 1:
            phi_txt = "—" if self._last_phi is None else f"{self._last_phi:.2f}"
            _log.info(
                f"  {'Manager':<14} {'control':<8} → {manager.active_backbone.model} "
                f"(rung {k}/{n - 1}, cost={manager.active_backbone.cost}, Φ_prev={phi_txt})"
            )
        return manager.active_backbone

    # ------------------------------------------------------------------ #
    # Outcome feedback (the manager's next-round risk signal)
    # ------------------------------------------------------------------ #
    def observe(self, phi: float) -> None:
        self._last_phi = phi
