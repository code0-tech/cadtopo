from typing import Dict, List

import numpy as np

from cadtopo.agent import Backbone
from cadtopo.manager import ManagerDecision
from cadtopo.orchestrator import CadTopoAI
from cadtopo.schema import AgentStep
from cadtopo.router import RoutingDecision


class FakeAgent:
    """Minimal stand-in for Agent: a scripted AgentStep per round.

    A round now has two phases: ``describe`` (phase 1, descriptors only) then
    ``step`` (phase 2, the work). ``describe`` returns neutral descriptors and
    does not touch memory; the scripted work steps drive ``step``. It carries a
    single-rung backbone ladder so the cost-aware selector can point at it.
    """

    def __init__(self, name: str, steps: List[AgentStep]):
        self.name = name
        self._steps = list(steps)
        self.calls: List[dict] = []
        self.describe_calls: List[dict] = []
        self.backbones = [Backbone(model="fake-model", cost=1.0)]
        self.active_backbone = self.backbones[0]

    def describe(self, task, round_goal, history=None):
        self.describe_calls.append({"task": task, "round_goal": round_goal, "history": list(history or [])})
        return AgentStep(query="", key="", accuracy=0.5)

    def step(self, task, round_goal, history=None):
        self.calls.append({"task": task, "round_goal": round_goal, "history": list(history or [])})
        return self._steps.pop(0)


class FakeRouter:
    """Coarse-selects every agent every round; topology is scripted per round."""

    def __init__(self, agents, decisions: List[RoutingDecision]):
        self.agents = agents
        self._decisions = list(decisions)
        self.coarse_select_calls: List[str] = []

    def coarse_select(self, text):
        self.coarse_select_calls.append(text)
        return list(self.agents), {a.name: 1.0 for a in self.agents}

    def induce_topology(self, active_agents, profiles):
        return self._decisions.pop(0)


class FakeManager:
    def __init__(self, decisions: List[ManagerDecision]):
        self._decisions = list(decisions)
        self.calls: List[dict] = []
        self.backbones = [Backbone(model="fake-manager-model", cost=1.0)]
        self.active_backbone = self.backbones[0]

    def review_round(self, **kwargs):
        self.calls.append(kwargs)
        return self._decisions.pop(0)


def _decision(agents, adjacency: np.ndarray | None = None) -> RoutingDecision:
    n = len(agents)
    return RoutingDecision(
        active_agents=agents,
        adjacency_matrix=adjacency if adjacency is not None else np.zeros((n, n), dtype=int),
        score_matrix=np.zeros((n, n)),
        execution_order=list(range(n)),
        profiles={},
    )


def _step(public="", accuracy=0.5) -> AgentStep:
    return AgentStep(public=public, query="", key="", accuracy=accuracy)


class TestHaltingAndDeliverable:
    def test_halts_on_first_round_when_manager_is_satisfied(self):
        agent = FakeAgent("Developer", [_step(public="the answer")])
        router = FakeRouter([agent], [_decision([agent])])
        manager = FakeManager([ManagerDecision(is_complete=True, phi=0.95, success_score=0.95, best_agent="Developer", best_result="the answer")])

        system = CadTopoAI(manager=manager, router=router, max_rounds=5)
        result = system.run("do the task")

        assert result == "the answer"
        assert len(agent.calls) == 1

    def test_round_1_runs_with_no_round_goal(self):
        agent = FakeAgent("Developer", [_step(public="x")])
        router = FakeRouter([agent], [_decision([agent])])
        manager = FakeManager([ManagerDecision(is_complete=True, phi=1.0, best_agent="Developer", best_result="x")])

        CadTopoAI(manager=manager, router=router, max_rounds=5).run("task")
        assert agent.calls[0]["round_goal"] == ""

    def test_next_goal_is_threaded_into_round_2(self):
        agent = FakeAgent("Developer", [_step(public="x"), _step(public="y")])
        router = FakeRouter([agent], [_decision([agent]), _decision([agent])])
        manager = FakeManager([
            ManagerDecision(is_complete=False, next_goal="Fix the bug.", phi=0.2, best_agent="Developer", best_result="x"),
            ManagerDecision(is_complete=True, phi=1.0, best_agent="Developer", best_result="y"),
        ])

        CadTopoAI(manager=manager, router=router, max_rounds=5).run("task")
        assert agent.calls[1]["round_goal"] == "Fix the bug."

    def test_returns_highest_phi_round_not_the_last_one(self):
        # Round 1 scores higher than round 2; the T_max cap must still return
        # round 1's result, not round 2's.
        agent = FakeAgent("Developer", [_step(public="r1"), _step(public="r2")])
        router = FakeRouter([agent], [_decision([agent]), _decision([agent])])
        manager = FakeManager([
            ManagerDecision(is_complete=False, next_goal="keep going", phi=0.9, best_agent="Developer", best_result="r1"),
            ManagerDecision(is_complete=False, next_goal="keep going", phi=0.3, best_agent="Developer", best_result="r2"),
        ])

        result = CadTopoAI(manager=manager, router=router, max_rounds=2).run("task")
        assert result == "r1"

    def test_equal_phi_ties_go_to_the_later_round(self):
        agent = FakeAgent("Developer", [_step(public="r1"), _step(public="r2")])
        router = FakeRouter([agent], [_decision([agent]), _decision([agent])])
        manager = FakeManager([
            ManagerDecision(is_complete=False, next_goal="keep going", phi=0.5, best_agent="Developer", best_result="r1"),
            ManagerDecision(is_complete=False, next_goal="keep going", phi=0.5, best_agent="Developer", best_result="r2"),
        ])

        result = CadTopoAI(manager=manager, router=router, max_rounds=2).run("task")
        assert result == "r2"

    def test_hits_round_cap_when_manager_never_halts(self):
        agent = FakeAgent("Developer", [_step(public="r1"), _step(public="r2"), _step(public="r3")])
        router = FakeRouter([agent], [_decision([agent]) for _ in range(3)])
        manager = FakeManager([
            ManagerDecision(is_complete=False, next_goal="g2", phi=0.1, best_agent="Developer", best_result="r1"),
            ManagerDecision(is_complete=False, next_goal="g3", phi=0.2, best_agent="Developer", best_result="r2"),
            ManagerDecision(is_complete=False, next_goal="g4", phi=0.3, best_agent="Developer", best_result="r3"),
        ])

        result = CadTopoAI(manager=manager, router=router, max_rounds=3).run("task")
        assert result == "r3"  # monotonically increasing phi -> last round wins
        assert len(agent.calls) == 3


class TestDeliverableFromManager:
    def test_orchestrator_returns_the_manager_chosen_deliverable(self):
        # The manager names the deliverable (Strong); the orchestrator no
        # longer pre-picks it from a coarse score.
        a = FakeAgent("Weak", [_step(public="weak output")])
        b = FakeAgent("Strong", [_step(public="strong output")])
        router = FakeRouter([a, b], [_decision([a, b])])
        manager = FakeManager([ManagerDecision(is_complete=True, phi=1.0, best_agent="Strong", best_result="strong output")])

        result = CadTopoAI(manager=manager, router=router, max_rounds=1).run("task")
        assert result == "strong output"

    def test_manager_receives_every_agent_output_and_no_pre_picked_deliverable(self):
        a = FakeAgent("A", [_step(public="out A")])
        b = FakeAgent("B", [_step(public="out B")])
        router = FakeRouter([a, b], [_decision([a, b])])
        manager = FakeManager([ManagerDecision(is_complete=True, phi=1.0, best_agent="A", best_result="out A")])

        CadTopoAI(manager=manager, router=router, max_rounds=1).run("task")
        call = manager.calls[0]
        assert call["results"] == {"A": "out A", "B": "out B"}
        # The orchestrator hands over every output and lets the manager choose.
        assert "best_agent" not in call
        assert "best_result" not in call

    def test_falls_back_to_aggregate_when_manager_names_no_deliverable(self):
        a = FakeAgent("A", [_step(public="out A")])
        router = FakeRouter([a], [_decision([a])])
        manager = FakeManager([ManagerDecision(is_complete=True, phi=1.0, best_agent="", best_result="")])

        result = CadTopoAI(manager=manager, router=router, max_rounds=1).run("task")
        assert "out A" in result


class TestMemoryUpdate:
    def test_forward_edge_routes_provider_output_into_consumer_same_round(self):
        # A -> B, and A runs before B in σ, so B conditions on A's CURRENT
        # round output — no one-round latency (the off-by-one fix).
        a = FakeAgent("A", [_step(public="pubA1")])
        b = FakeAgent("B", [_step(public="pubB1")])

        adjacency = np.array([[0, 1], [0, 0]])  # A -> B, execution order [A, B]
        router = FakeRouter([a, b], [_decision([a, b], adjacency=adjacency)])
        manager = FakeManager([ManagerDecision(is_complete=True, phi=1.0, best_agent="A", best_result="pubA1")])

        CadTopoAI(manager=manager, router=router, max_rounds=5).run("task")

        # B's FIRST (and only) pass already sees A's round-1 public output.
        history = b.calls[0]["history"]
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert "pubA1" in history[0]["content"]
        assert "from A" in history[0]["content"]

    def test_handoff_routes_the_full_public_output_same_round(self):
        # A produces code in PUBLIC; the consumer gets that code deterministically
        # (every deliverable is public now — there is no private channel).
        a = FakeAgent("A", [_step(public="```python\ndef f(): pass\n```")])
        b = FakeAgent("B", [_step(public="pubB1")])

        adjacency = np.array([[0, 1], [0, 0]])  # A -> B
        router = FakeRouter([a, b], [_decision([a, b], adjacency=adjacency)])
        manager = FakeManager([ManagerDecision(is_complete=True, phi=1.0, best_agent="A", best_result="x")])

        CadTopoAI(manager=manager, router=router, max_rounds=5).run("task")

        routed = b.calls[0]["history"][0]["content"]
        assert "def f(): pass" in routed

    def test_back_edge_from_a_later_agent_is_deferred_to_next_round(self):
        # B -> A, but the execution order is [A, B], so B (the provider) runs
        # AFTER A (the consumer). A cannot see B in-round; B's output reaches
        # A only in the next round's memory.
        a = FakeAgent("A", [_step(public="pubA1"), _step(public="pubA2")])
        b = FakeAgent("B", [_step(public="pubB1"), _step(public="pubB2")])

        adjacency = np.array([[0, 0], [1, 0]])  # B -> A, order [A, B]
        router = FakeRouter([a, b], [_decision([a, b], adjacency=adjacency), _decision([a, b])])
        manager = FakeManager([
            ManagerDecision(is_complete=False, next_goal="g2", phi=0.1, best_agent="A", best_result="pubA1"),
            ManagerDecision(is_complete=True, phi=1.0, best_agent="A", best_result="pubA2"),
        ])

        CadTopoAI(manager=manager, router=router, max_rounds=5).run("task")

        # Round 1 A saw nothing routed (B ran after it); round 2 A finally has B's note.
        assert a.calls[0]["history"] == []
        history = a.calls[1]["history"]
        assert history[0] == {"role": "assistant", "content": "pubA1"}
        assert "pubB1" in history[1]["content"]
        assert "from B" in history[1]["content"]
        assert history[1]["role"] == "user"

    def test_no_active_edge_means_no_message_is_routed(self):
        a = FakeAgent("A", [_step(public="pubA1")])
        b = FakeAgent("B", [_step(public="pubB1")])

        router = FakeRouter([a, b], [_decision([a, b])])  # zero adjacency
        manager = FakeManager([ManagerDecision(is_complete=True, phi=1.0, best_agent="A", best_result="pubA1")])

        CadTopoAI(manager=manager, router=router, max_rounds=5).run("task")
        assert b.calls[0]["history"] == []  # nothing routed in


class TestTwoPhaseRound:
    def test_descriptors_are_collected_before_work_and_feed_the_topology(self):
        # Phase 1: describe() runs for every active agent and its query/key
        # feed induce_topology; only THEN do the phase-2 work passes run.
        events: List[tuple] = []

        class TracingAgent(FakeAgent):
            def describe(self, task, round_goal, history=None):
                events.append(("describe", self.name))
                return AgentStep(query=f"{self.name}-q", key=f"{self.name}-k", accuracy=0.5)

            def step(self, task, round_goal, history=None):
                events.append(("step", self.name))
                return super().step(task, round_goal, history)

        a = TracingAgent("A", [_step(public="pa")])
        b = TracingAgent("B", [_step(public="pb")])

        captured: Dict[str, dict] = {}

        class CapturingRouter(FakeRouter):
            def induce_topology(self, active_agents, profiles):
                captured["profiles"] = profiles
                return super().induce_topology(active_agents, profiles)

        router = CapturingRouter([a, b], [_decision([a, b])])
        manager = FakeManager([ManagerDecision(is_complete=True, phi=1.0, best_agent="A", best_result="pa")])
        CadTopoAI(manager=manager, router=router, max_rounds=1).run("task")

        # Every describe precedes every step (topology is fixed before work).
        assert events == [("describe", "A"), ("describe", "B"), ("step", "A"), ("step", "B")]
        # The topology was induced from exactly the describe() descriptors.
        assert captured["profiles"]["A"] == {"key": "A-k", "query": "A-q", "accuracy": 0.5}
        assert captured["profiles"]["B"] == {"key": "B-k", "query": "B-q", "accuracy": 0.5}

    def test_own_memory_turn_is_the_clean_deliverable_without_tool_breadcrumb(self):
        step_with_tools = _step(public="the code")
        step_with_tools.tool_calls = [{"name": "check_python_syntax", "arguments": "{}", "result": "OK"}]
        a = FakeAgent("A", [step_with_tools, _step(public="the code v2")])
        router = FakeRouter([a], [_decision([a]), _decision([a])])
        manager = FakeManager([
            ManagerDecision(is_complete=False, next_goal="g2", phi=0.1, best_agent="A", best_result="the code"),
            ManagerDecision(is_complete=True, phi=1.0, best_agent="A", best_result="the code v2"),
        ])

        result = CadTopoAI(manager=manager, router=router, max_rounds=5).run("task")

        assert result == "the code v2"
        # A's own memory turn for round 1 is EXACTLY its deliverable — no tool
        # breadcrumb glued on (weak models echo such text back into their next
        # answer, corrupting it).
        own_turn = a.calls[1]["history"][0]
        assert own_turn["role"] == "assistant"
        assert own_turn["content"] == "the code"
        assert "check_python_syntax" not in own_turn["content"]
