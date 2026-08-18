"""The CADTopo router: coarse agent selection + per-round topology induction."""

from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np

from .agent import Agent
from .embedding import EmbeddingModel
from .logging_utils import get_logger

_log = get_logger("cadtopo.router")


def _shorten(text: str, limit: int = 70) -> str:
    """Trim a single-line preview of a long descriptor, for log output."""
    text = (text or "").strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


class RoutingDecision(NamedTuple):
    """Everything one round of routing produces.

    ``execution_order`` is the DyTopo aggregation order σ(t) (Sec. 3.4): a
    permutation of indices into ``active_agents`` such that providers come
    before their consumers whenever the topology allows it.
    """

    active_agents: List[Agent]
    adjacency_matrix: np.ndarray
    score_matrix: np.ndarray
    execution_order: List[int]
    profiles: Dict[str, Dict[str, Any]]


def compute_execution_order(adjacency_matrix: np.ndarray) -> Tuple[List[int], bool]:
    """Adaptive Topological Sequencing (DyTopo Sec. 3.4.2, Eq. 10/11).

    Iteratively selects the unplaced node with the smallest restricted
    in-degree (in-degree counted only within the remaining sub-graph), with
    deterministic index tie-breaking. On a DAG this yields a valid
    topological order; on a cyclic graph it greedily breaks cycles so
    reliance on cyclically unavailable information is minimised.

    :return: (σ as a list of node indices, True if a cycle had to be broken)
    """
    n = adjacency_matrix.shape[0]
    unplaced = list(range(n))
    order: List[int] = []
    cycle_broken = False
    while unplaced:
        restricted_in_degree = {
            i: int(sum(adjacency_matrix[j, i] for j in unplaced if j != i))
            for i in unplaced
        }
        best = min(unplaced, key=lambda i: (restricted_in_degree[i], i))
        if restricted_in_degree[best] > 0:
            cycle_broken = True
        order.append(best)
        unplaced.remove(best)
    return order, cycle_broken


class Router:
    """Static coarse selection (Stage 1) + dynamic topology induction (Stage 2/3).

    :param agents: The global agent pool.
    :param embedding_model: Embedding backend for skill/goal/query/key matching.
    :param theta: Gating threshold (θ) for Stage 1 coarse selection.
    :param tau: Hard threshold (τ) on the routing score for Stage 2 edges.
    """

    def __init__(
        self,
        agents: List[Agent],
        embedding_model: EmbeddingModel,
        theta: float = 0.5,
        tau: float = 0.3,
    ):
        self.agents = agents
        self.embedder = embedding_model
        self.theta = theta
        self.tau = tau

        # Pre-vectorise every agent's static skill definition up front.
        self.agent_embeddings = {
            agent.name: self.embedder.get_embedding(agent.skill_definition)
            for agent in self.agents
        }

    def coarse_select(self, round_goal: str) -> Tuple[List[Agent], Dict[str, float]]:
        """STAGE 1 — static pre-filtering, no LLM calls.

        Selects the agents that participate this round by matching each
        agent's static skill embedding against the round goal (θ gate). Runs
        BEFORE the agents' single passes, so a forward pass is only spent on
        relevant agents. Never returns an empty set: on a total miss it keeps
        the single best-matching agent.

        Agents flagged ``mandatory`` (see :class:`~cadtopo.agent.Agent`)
        bypass the θ gate and are always active; their real relevance is
        still computed and reported, so ranking (e.g. the deliverable pick)
        stays honest.

        :return: ``(active_agents, relevance_by_name)`` — the second element
            maps EVERY agent's name to its coarse relevance, so the caller
            can rank the round's outputs by coarse relevance too.
        """
        if not self.agents:
            return [], {}

        g_t = self.embedder.get_query_embedding(round_goal)
        relevances = [
            (self.embedder.cosine_similarity(g_t, self.agent_embeddings[agent.name]), agent)
            for agent in self.agents
        ]
        relevance_by_name = {agent.name: rel for rel, agent in relevances}

        passed = [
            (rel, a) for rel, a in relevances
            if rel >= self.theta or getattr(a, "mandatory", False)
        ]
        if passed:
            active_agents = [a for _, a in passed]
            ranked = sorted(passed, key=lambda x: -x[0])
            picks = ", ".join(
                f"{a.name}({rel:.2f}{', mandatory' if getattr(a, 'mandatory', False) and rel < self.theta else ''})"
                for rel, a in ranked
            )
            _log.info(f"Coarse: {len(active_agents)}/{len(self.agents)} above θ={self.theta} or mandatory — {picks}")
        else:
            best_rel, best_agent = max(relevances, key=lambda x: x[0])
            active_agents = [best_agent]
            _log.warning(
                f"Coarse: 0/{len(self.agents)} above θ={self.theta}; "
                f"fallback to best match {best_agent.name}({best_rel:.3f})"
            )
        return active_agents, relevance_by_name

    def induce_topology(
        self,
        active_agents: List[Agent],
        profiles: Dict[str, Dict[str, Any]],
    ) -> RoutingDecision:
        """STAGE 2/3 — induce the round's directed topology from the agents'
        own descriptors (Sec. 3.3), then order it (Sec. 3.4).

        Makes NO LLM calls: the query/key descriptors and self-assessed
        accuracy come from each agent's phase-1 :meth:`Agent.describe`
        call, which ran this round BEFORE any work. The edges and aggregation
        order σ(t) induced here govern THIS round's work phase: agents run in
        σ(t) order, and a provider's fresh output is routed to its downstream
        consumers within the same round (forward edges), with back edges
        deferred to the next round's memory (see the orchestrator).

        :param active_agents: Agents that produced a pass this round.
        :param profiles: ``name -> {"query": str, "key": str, "accuracy": float}``.
        """
        if not active_agents:
            return RoutingDecision([], np.zeros((0, 0)), np.zeros((0, 0)), [], {})

        dynamic_profiles = self._embed_profiles(active_agents, profiles)
        score_matrix = self._score_matrix(active_agents, dynamic_profiles)

        adjacency_matrix = np.where(score_matrix >= self.tau, 1, 0)
        np.fill_diagonal(adjacency_matrix, 0)
        self._log_topology(active_agents, adjacency_matrix)

        execution_order, cycle_broken = compute_execution_order(adjacency_matrix)
        route = " → ".join(active_agents[i].name for i in execution_order)
        _log.info(
            f"Execution order: {route}" + (" (cyclic topology — greedy cycle-breaking)" if cycle_broken else "")
        )

        return RoutingDecision(
            active_agents=active_agents,
            adjacency_matrix=adjacency_matrix,
            score_matrix=score_matrix,
            execution_order=execution_order,
            profiles=dynamic_profiles,
        )

    def _embed_profiles(
        self, active_agents: List[Agent], profiles: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Embed this round's key/query descriptors and log them."""
        dynamic_profiles: Dict[str, Dict[str, Any]] = {}
        for agent in active_agents:
            p = profiles.get(agent.name, {})
            key = p.get("key", "") or ""
            query = p.get("query", "") or ""
            accuracy = float(p.get("accuracy", 0.5))
            dynamic_profiles[agent.name] = {
                "key": key,
                "query": query,
                "key_emb": self.embedder.get_embedding(key),
                "query_emb": self.embedder.get_query_embedding(query),
                "accuracy": accuracy,
            }
            _log.info(
                f"  {agent.name:<14} Acc={accuracy:.2f} Cost={agent.cost} "
                f"Q={_shorten(query)!r} K={_shorten(key)!r}"
            )
        return dynamic_profiles

    @staticmethod
    def _score_matrix(
        active_agents: List[Agent],
        dynamic_profiles: Dict[str, Dict[str, Any]],
    ) -> np.ndarray:
        """Pairwise routing score S_ij = fit(i→j): the cosine similarity between
        provider i's key descriptor and consumer j's query descriptor. Topology
        is driven purely by descriptor fit — there is no accuracy or cost term.
        """
        n = len(active_agents)
        score_matrix = np.zeros((n, n))
        for i, agent_i in enumerate(active_agents):
            k_i = dynamic_profiles[agent_i.name]["key_emb"]
            for j, agent_j in enumerate(active_agents):
                if i == j:
                    continue
                q_j = dynamic_profiles[agent_j.name]["query_emb"]
                score_matrix[i, j] = EmbeddingModel.cosine_similarity(k_i, q_j)
        return score_matrix

    @staticmethod
    def _log_topology(active_agents: List[Agent], adjacency_matrix: np.ndarray) -> None:
        """Log the actual data path: per agent, its incoming sources."""
        n = len(active_agents)
        path_parts = []
        any_edges = False
        for j, agent_j in enumerate(active_agents):
            incoming = [active_agents[i].name for i in range(n) if adjacency_matrix[i, j] == 1]
            if incoming:
                any_edges = True
                path_parts.append(f"{agent_j.name}←{','.join(incoming)}")
            else:
                path_parts.append(agent_j.name)
        suffix = "" if any_edges else ", no data exchange"
        _log.info(f"Topology: {n} active{suffix} — {' | '.join(path_parts)}")
