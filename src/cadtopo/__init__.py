"""CADTopo — cost-aware dynamic-topology multi-agent LLM orchestration.

An implementation of DyTopo's protocol (per-round agent selection, dynamic
topology induction from agents' own offer/demand descriptors, and a
meta-agent that scores each round and decides whether to halt), plus a
cost-aware routing extension. See ``README.md`` for the full architecture.

Typical usage::

    from cadtopo import (
        CadTopoAI, Router, Manager, Agent, EmbeddingModel,
    )

    agents = [Agent(name="Developer", skill_definition="...", ...), ...]
    router = Router(agents=agents, embedding_model=EmbeddingModel())
    manager = Manager(api_provider="openrouter/...", api_key="...")
    system = CadTopoAI(manager=manager, router=router)

    answer = system.run("Implement a function that reverses a string.")
"""

from .agent import Agent
from .backbone import Backbone
from .embedding import EmbeddingModel
from .manager import CompletedSubTask, ManagerDecision, Manager, aggregate_public
from .orchestrator import AgentMemory, CadTopoAI
from .router import Router, RoutingDecision, compute_execution_order
from .schema import AgentStep, Descriptors, RoundDecision, Tool
from .selection import CostAwareSelector, ModelSelector

__all__ = [
    "CadTopoAI",
    "Router",
    "RoutingDecision",
    "compute_execution_order",
    "Manager",
    "ManagerDecision",
    "CompletedSubTask",
    "aggregate_public",
    "Agent",
    "Backbone",
    "AgentStep",
    "Descriptors",
    "RoundDecision",
    "Tool",
    "AgentMemory",
    "EmbeddingModel",
    "CostAwareSelector",
    "ModelSelector",
]

__version__ = "0.1.0"
