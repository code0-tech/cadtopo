import numpy as np

from cadtopo.router import Router, compute_execution_order
from cadtopo.agent import Agent
from tests.conftest import FakeEmbedder


def _agent(name: str, skill: str = "", cost: float = 1.0, mandatory: bool = False) -> Agent:
    return Agent(name=name, skill_definition=skill, cost_per_token=cost, api_provider="test-model", mandatory=mandatory)


class TestComputeExecutionOrder:
    def test_dag_yields_topological_order(self):
        # 0 -> 1 -> 2
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        order, cycle_broken = compute_execution_order(adj)
        assert order == [0, 1, 2]
        assert cycle_broken is False

    def test_no_edges_falls_back_to_index_order(self):
        adj = np.zeros((3, 3), dtype=int)
        order, cycle_broken = compute_execution_order(adj)
        assert order == [0, 1, 2]
        assert cycle_broken is False

    def test_cycle_is_broken_deterministically(self):
        # 0 -> 1 -> 0 (a 2-cycle)
        adj = np.array([[0, 1], [1, 0]])
        order, cycle_broken = compute_execution_order(adj)
        assert cycle_broken is True
        assert sorted(order) == [0, 1]

    def test_diamond_topology(self):
        # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        adj = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
        order, cycle_broken = compute_execution_order(adj)
        assert cycle_broken is False
        assert order.index(0) < order.index(1) < order.index(3)
        assert order.index(0) < order.index(2) < order.index(3)


class TestCoarseSelect:
    def test_keeps_agents_above_theta(self):
        goal_vec = np.array([1.0, 0.0])
        agents = [_agent("A"), _agent("B"), _agent("C")]
        embedder = FakeEmbedder({
            "goal": goal_vec,
            "": np.array([1.0, 0.0]),  # every agent's empty skill_definition maps here
        })
        router = Router(agents=agents, embedding_model=embedder, theta=0.5)
        active, relevances = router.coarse_select("goal")
        assert {a.name for a in active} == {"A", "B", "C"}
        assert set(relevances) == {"A", "B", "C"}

    def test_falls_back_to_best_match_when_nothing_clears_theta(self):
        embedder = FakeEmbedder({
            "goal": np.array([1.0, 0.0]),
            "skill-a": np.array([0.9, 0.1]),
            "skill-b": np.array([-1.0, 0.0]),  # opposite direction: worst match
        })
        agents = [_agent("A", skill="skill-a"), _agent("B", skill="skill-b")]
        router = Router(agents=agents, embedding_model=embedder, theta=0.99)
        active, relevances = router.coarse_select("goal")
        assert len(active) == 1
        assert active[0].name == "A"
        assert len(relevances) == 2  # relevance is still reported for every agent

    def test_empty_agent_pool_returns_empty(self):
        router = Router(agents=[], embedding_model=FakeEmbedder())
        active, relevances = router.coarse_select("goal")
        assert active == []
        assert relevances == {}

    def test_mandatory_agent_bypasses_theta_gate(self):
        embedder = FakeEmbedder({
            "goal": np.array([1.0, 0.0]),
            "skill-a": np.array([0.9, 0.1]),
            "skill-b": np.array([-1.0, 0.0]),  # worst possible match
        })
        agents = [_agent("A", skill="skill-a"), _agent("B", skill="skill-b", mandatory=True)]
        router = Router(agents=agents, embedding_model=embedder, theta=0.5)
        active, relevances = router.coarse_select("goal")
        assert {a.name for a in active} == {"A", "B"}
        # The mandatory agent's REAL relevance is still reported (not faked
        # above θ), so deliverable ranking by coarse relevance stays honest.
        assert relevances["B"] < router.theta

    def test_mandatory_only_round_does_not_hit_the_fallback(self):
        # Nothing clears θ, but a mandatory agent exists: the active set is
        # exactly the mandatory agent, not the best-match fallback.
        embedder = FakeEmbedder({
            "goal": np.array([1.0, 0.0]),
            "skill-a": np.array([0.0, 1.0]),  # orthogonal: below θ
            "skill-b": np.array([-1.0, 0.0]),
        })
        agents = [_agent("A", skill="skill-a"), _agent("B", skill="skill-b", mandatory=True)]
        router = Router(agents=agents, embedding_model=embedder, theta=0.9)
        active, _ = router.coarse_select("goal")
        assert [a.name for a in active] == ["B"]


class TestInduceTopology:
    def test_no_active_agents_returns_empty_decision(self):
        router = Router(agents=[], embedding_model=FakeEmbedder())
        decision = router.induce_topology([], {})
        assert decision.active_agents == []
        assert decision.execution_order == []

    def test_high_fit_produces_an_edge(self):
        # A offers exactly what B demands, both perfectly aligned vectors.
        embedder = FakeEmbedder({
            "offer": np.array([1.0, 0.0]),
            "demand": np.array([1.0, 0.0]),
            "": np.zeros(2),
        })
        a = _agent("A")
        b = _agent("B")
        router = Router(agents=[a, b], embedding_model=embedder, tau=0.5)

        profiles = {
            "A": {"key": "offer", "query": "", "accuracy": 1.0},
            "B": {"key": "", "query": "demand", "accuracy": 1.0},
        }
        decision = router.induce_topology([a, b], profiles)

        i, j = 0, 1  # A -> B
        assert decision.adjacency_matrix[i, j] == 1
        assert decision.adjacency_matrix[j, i] == 0  # B doesn't offer what A wants (zero vector)

    def test_low_fit_suppresses_an_edge(self):
        # A's offer is orthogonal to B's demand: fit below tau, no edge.
        embedder = FakeEmbedder({"offer": np.array([1.0, 0.0]), "demand": np.array([0.0, 1.0]), "": np.zeros(2)})
        a = _agent("A")
        b = _agent("B")
        router = Router(agents=[a, b], embedding_model=embedder, tau=0.5)

        profiles = {
            "A": {"key": "offer", "query": "", "accuracy": 1.0},
            "B": {"key": "", "query": "demand", "accuracy": 1.0},
        }
        decision = router.induce_topology([a, b], profiles)
        assert decision.adjacency_matrix[0, 1] == 0

    def test_diagonal_is_always_zero(self):
        embedder = FakeEmbedder({"": np.array([1.0, 0.0])})
        a = _agent("A")
        router = Router(agents=[a], embedding_model=embedder, tau=-999.0)  # force every score to pass
        decision = router.induce_topology([a], {"A": {"key": "x", "query": "x", "accuracy": 1.0}})
        assert decision.adjacency_matrix[0, 0] == 0

    def test_missing_profile_defaults_to_neutral_accuracy(self):
        embedder = FakeEmbedder({"": np.zeros(2)})
        a, b = _agent("A"), _agent("B")
        router = Router(agents=[a, b], embedding_model=embedder)
        # No profile entry at all for either agent — must not raise.
        decision = router.induce_topology([a, b], {})
        assert decision.profiles["A"]["accuracy"] == 0.5
