"""The CADTopo system: orchestrates the full DyTopo protocol flow (Sec. 3.2–3.5)."""

from dataclasses import dataclass, field
from typing import Dict, List

from .agent import AgentStep, Agent
from .logging_utils import get_logger
from .manager import CompletedSubTask, Manager, aggregate_public
from .router import Router, RoutingDecision
from .selection import CostAwareSelector, ModelSelector
from .telemetry import COST

_log = get_logger("cadtopo.orchestrator")


@dataclass
class AgentMemory:
    """Per-agent memory H_i (DyTopo Eq. 3).

    ``turns`` is the agent's FULL accumulated memory across all past
    rounds — a real chat transcript, not a single-round window. Each round
    appends, in σ(t) order: the fresh hand-offs routed in from upstream
    providers that already ran THIS round (``user`` turns), the agent's own
    public result (an ``assistant`` turn), then the hand-offs from providers
    that ran AFTER it this round (``user`` turns, folded at round end).
    Empty in round 1.
    """

    turns: List[Dict[str, str]] = field(default_factory=list)


class CadTopoAI:
    """Orchestrates the full DyTopo protocol flow (Sec. 3.2–3.5).

    Each round runs in TWO phases so the topology can route WITHIN the round
    (the CADTopo fix for DyTopo's one-round routing latency):

      1. Round 1 runs with NO round goal — a CADTopo deviation from vanilla
         DyTopo, which has the manager set every goal including the first
         (:meth:`Manager.initial_goal`, still available but not called
         from here). Agents see only the constant task for round 1; the
         manager starts steering from round 2 onward via
         :meth:`Manager.review_round`'s ``next_goal``. C_task is TWO
         layers: the constant task (pinned into every agent's context every
         round) plus the round goal (a per-round directive layered on it,
         ``""`` in round 1).
      2. Each round:
         a. Coarse selection picks the participating agents (router Stage 1),
            matched against the task + the round goal.
         b. PHASE 1 — descriptors (Sec. 3.3). Every active agent runs a
            lightweight :meth:`Agent.describe` call that returns ONLY its
            query/key/accuracy for this round (no work, no tools). These are
            causally independent, so the order does not matter.
         c. The router induces THIS round's directed topology G(t) and
            aggregation order σ(t) from those descriptors (Sec. 3.3/3.4) —
            before any work is done.
         d. PHASE 2 — work (Sec. 3.2.1). The active agents run their single
            work passes SEQUENTIALLY in σ(t) order. As each agent is about to
            run, the FRESH current-round hand-offs from its upstream
            providers that have already run this round (forward edges of
            G(t)) are folded into its memory first — so a consumer conditions
            on this round's provider output, not last round's. This is
            CADTopo's intra-round routing; it replaces vanilla DyTopo's strict
            synchronisation barrier and is what lets e.g. the Tester execute
            the SAME code the manager will then score (closing the
            off-by-one).
         e. MEMORY UPDATE (Eq. 3) — the BACK edges of G(t) are folded in: for
            every agent, the hand-offs from its providers that ran AFTER it
            this round (cyclically unavailable in-round) are appended to H_i,
            so they reach it next round. Every active edge is thus delivered
            exactly once — forward edges in-round, back edges next round.
         f. The manager reads the global state (round goal + this round's
            public messages, Eq. 12) in ONE execution-free pass and CHOOSES
            the round's deliverable itself — the agent whose output is the
            finished artifact answering the request (it names BEST_AGENT),
            never an evaluator's verdict or critique. In the same pass it
            scores that deliverable (the model reports failure_score = P(wrong);
            Φ = 1 - failure_score ∈ [0,1]; halting y(t) is derived by
            thresholding Φ ≥ γ_success, Eq. 14) and drafts the
            next goal C_task(t+1) (Eq. 13). On halt the next goal is ignored.
      3. On halt (or the round cap T_max), the deliverable is the highest-Φ
         round's manager-chosen best-agent result (falling back to the
         ordered concatenation of that round's public messages, Eq. 12, only
         if the manager named no usable agent). The same read-out applies
         whether the manager halted early or the cap was reached.

    :param manager: The central manager agent.
    :param router: The router (coarse selection + topology induction).
    :param max_rounds: Hard upper bound on rounds — the T_max infinite-loop guard.
    :param selector: The cost-aware model selector (see
        :mod:`cadtopo.selection`) that picks each agent's backbone per round.
        Defaults to a :class:`~cadtopo.selection.CostAwareSelector`. With
        single-backbone agents it is a no-op (only one rung to pick).
    """

    def __init__(
        self,
        manager: Manager,
        router: Router,
        max_rounds: int = 5,
        selector: ModelSelector | None = None,
    ):
        self.manager = manager
        self.router = router
        self.max_rounds = max_rounds
        self.selector = selector or CostAwareSelector()
        # Wire the selector to the live manager + agent pool so it knows every
        # ladder and can drive the joint manager-first escalation policy.
        self.selector.bind(manager, router.agents)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, user_prompt: str) -> str:
        _log.debug(f"Run started — request: {user_prompt[:120]!r}")

        # Independent task: start from clean ladders so rungs never leak between
        # runs (the selector is reused across a whole benchmark).
        self.selector.reset()

        completed: List[CompletedSubTask] = []
        # Highest-Φ round seen so far. Φ can regress between rounds (a
        # refinement can make the answer worse), so the T_max read-out
        # returns this best round rather than whatever the last round
        # happened to produce.
        best_round: CompletedSubTask | None = None
        best_phi = -1.0

        task = user_prompt
        round_goal = ""  # round 1 has no goal yet — see class docstring
        memory: Dict[str, AgentMemory] = {agent.name: AgentMemory() for agent in self.router.agents}

        for t in range(1, self.max_rounds + 1):
            cost_before_round = COST.snapshot()
            goal_desc = round_goal[:120] if round_goal else "<none — task only, round 1>"
            _log.info(f"Round {t}/{self.max_rounds} — goal: {goal_desc!r}")

            active_agents, _coarse_scores = self.router.coarse_select(
                f"{task}\n\n{round_goal}" if round_goal else task
            )

            # PHASE 1 — descriptors: induce this round's topology + σ(t) from
            # the agents' own query/key/confidence, before any work is done. The
            # cheap descriptor pass runs on each agent's cheapest backbone.
            for a in active_agents:
                self.selector.for_describe(a)
            descriptors = {a.name: a.describe(task, round_goal, memory[a.name].turns) for a in active_agents}

            # Point each agent at its current WORK rung (set by the joint
            # policy's Φ history; no per-round bump). Done before topology
            # induction so the chosen backbone's cost is the Cost_i the router
            # scores this round.
            for a in active_agents:
                self.selector.for_work(a, confidence=descriptors[a.name].accuracy, round_num=t, max_rounds=self.max_rounds)

            decision = self.router.induce_topology(
                active_agents=active_agents,
                profiles={
                    a.name: {"key": descriptors[a.name].key, "query": descriptors[a.name].query, "accuracy": descriptors[a.name].accuracy}
                    for a in active_agents
                },
            )

            # PHASE 2 — work: run in σ(t) order with fresh forward hand-offs
            # routed in-round; then fold the back edges into memory for next
            # round.
            order = [decision.active_agents[i] for i in decision.execution_order]
            steps = self._run_passes(order, task, round_goal, memory, decision, round_num=t)
            self._fold_back_edges(memory, order, decision, steps, round_num=t)
            self._log_round_cost(t, cost_before_round)

            public = {
                decision.active_agents[j].name: steps[decision.active_agents[j].name].public
                for j in decision.execution_order
            }

            # Point the manager at its control rung — its risk is how far the
            # PREVIOUS round's Φ fell short — before it reads the round.
            self.selector.for_control(self.manager, round_num=t, max_rounds=self.max_rounds)

            # The manager reads every agent's output and CHOOSES the round's
            # deliverable itself (it names BEST_AGENT), then scores it — the
            # deliverable is no longer fixed by a router coarse score.
            control = self.manager.review_round(
                user_prompt=user_prompt, round_goal=round_goal, results=public,
                completed=completed, round_num=t, max_rounds=self.max_rounds,
            )
            round_state = CompletedSubTask(goal=round_goal, result=control.best_result or aggregate_public(public))
            completed.append(round_state)

            # Record this round's Φ — it is the manager's risk signal next round.
            self.selector.observe(control.phi)

            # Ties go to the LATER round: with equal Φ, the more-refined round wins.
            if control.phi >= best_phi:
                best_phi = control.phi
                best_round = round_state

            if control.is_complete or t == self.max_rounds:
                reason = "Manager halted (Π_meta)" if control.is_complete else f"Round cap T_max={self.max_rounds} reached"
                _log.info(f"{reason} after round {t}; returning best round (Φ={best_phi:.3f}).")
                return best_round.result

            round_goal = control.next_goal

        return ""  # unreachable: the loop always returns at t == max_rounds

    @staticmethod
    def _log_round_cost(round_num: int, baseline_snapshot) -> None:
        delta = COST.diff_since(baseline_snapshot)
        agg = COST.aggregate(delta)
        if agg.calls == 0:
            return
        _log.info(f"Round {round_num} cost: {agg.calls} calls / {agg.total_tokens:,} tokens (cum: {COST.total.total_tokens:,})")

    # ------------------------------------------------------------------ #
    # Phase 2 — work passes (Sec. 3.2.1) + intra-round routing (forward edges)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _run_passes(
        order: List[Agent],
        task: str,
        round_goal: str,
        memory: Dict[str, AgentMemory],
        decision: RoutingDecision,
        round_num: int,
    ) -> Dict[str, AgentStep]:
        """Run one work pass per active agent, in σ(t) order, routing in-round.

        Each agent conditions on the constant task, its own accumulated
        memory H_i, and the round goal. Before an agent runs, the FRESH
        current-round hand-offs from its upstream providers that have already
        run this round (forward edges of THIS round's topology G(t)) are
        folded into H_i — so it sees this round's provider output, not last
        round's. This is CADTopo's intra-round routing (it replaces vanilla
        DyTopo's synchronisation barrier). Providers that run AFTER the
        consumer this round (back edges) are folded in later by
        :meth:`_fold_back_edges`, reaching the consumer next round.

        Follows DyTopo's memory update (Eq. 3): H_i (``mem.turns``)
        accumulates across ALL rounds. Here we append the incoming forward
        hand-offs (``user`` turns) then the agent's own public contribution
        (an ``assistant`` turn) for this round.

        Tool-calling (CADTopo extension, not in DyTopo): a pass with tools
        may internally run several LLM↔tool round-trips (see
        :meth:`Agent.step`), but only the FINAL plain-text deliverable — plus a
        deterministic breadcrumb of which tools were already tried — is folded
        into H_i.
        """
        active_by_name = {a.name: a for a in order}
        steps: Dict[str, AgentStep] = {}
        for agent in order:
            mem = memory[agent.name]
            # Forward edges: providers that already ran this round hand off
            # their FRESH output before this agent conditions on its memory.
            for provider in CadTopoAI._providers_for(agent.name, active_by_name, decision):
                if provider.name not in steps:
                    continue  # provider runs after this agent — a back edge, folded later
                content = steps[provider.name].public
                if content:
                    mem.turns.append(
                        {"role": "user", "content": f"[Round {round_num}] hand-off from {provider.name}:\n{content}"}
                    )

            _log.debug(f"Pass: {agent.name}")
            step = agent.step(task, round_goal, mem.turns)
            steps[agent.name] = step

            # The agent's own deliverable is its memory turn — clean, with no
            # tool-call breadcrumb glued on (weak models echo such breadcrumbs
            # straight back into their next deliverable, corrupting it).
            mem.turns.append({"role": "assistant", "content": step.public})
        return steps

    # ------------------------------------------------------------------ #
    # Memory update (DyTopo Eq. 3) — back edges of G(t)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fold_back_edges(
        memory: Dict[str, AgentMemory],
        order: List[Agent],
        decision: RoutingDecision,
        steps: Dict[str, AgentStep],
        round_num: int,
    ) -> None:
        """Fold this round's BACK edges into memory for next round (Eq. 3).

        An edge j→i where provider j ran AFTER consumer i this round could
        not be delivered in-round (i had already run), so its fresh output is
        appended to H_i now, as a persistent ``user`` turn — reaching i next
        round. Forward edges (j ran before i) were already delivered in-round
        by :meth:`_run_passes`; every active edge is thus folded exactly once.
        WHAT j hands off is its full public deliverable (there is no private
        channel), routed deterministically here.
        """
        active_by_name = {a.name: a for a in order}
        pos = {a.name: i for i, a in enumerate(order)}
        for consumer in order:
            folded = []
            for provider in CadTopoAI._providers_for(consumer.name, active_by_name, decision):
                if pos[provider.name] <= pos[consumer.name]:
                    continue  # forward edge — already delivered in-round
                content = steps[provider.name].public
                if content:
                    memory[consumer.name].turns.append(
                        {"role": "user", "content": f"[Round {round_num}] hand-off from {provider.name}:\n{content}"}
                    )
                    folded.append(provider.name)
            if folded:
                _log.debug(f"Memory (deferred to next round): {consumer.name} ← {', '.join(folded)}")

    @staticmethod
    def _providers_for(
        consumer_name: str,
        active_by_name: Dict[str, Agent],
        decision: RoutingDecision,
    ) -> List[Agent]:
        """Upstream providers feeding ``consumer_name`` this round, restricted
        to active agents and ordered by σ(t) (so hand-offs fold in aggregation
        order, Eq. 3).

        Edge j→i active (``adjacency[j, i] == 1``) means agent j is a
        provider for agent i.
        """
        names = [a.name for a in decision.active_agents]
        if consumer_name not in names or consumer_name not in active_by_name:
            return []
        ci = names.index(consumer_name)
        adj = decision.adjacency_matrix
        sigma_pos = {idx: pos for pos, idx in enumerate(decision.execution_order)}
        providers = [
            (sigma_pos.get(pj, pj), names[pj])
            for pj in range(len(names))
            if pj != ci and adj[pj, ci] == 1 and names[pj] in active_by_name
        ]
        providers.sort()
        return [active_by_name[name] for _, name in providers]
