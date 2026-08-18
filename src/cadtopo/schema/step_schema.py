from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AgentStep:
    """One agent's output for a round (DyTopo Eq. 2).

    ``query``/``key``/``accuracy`` come from the phase-1
    :meth:`cadtopo.agent.Agent.describe` call; ``public`` is the phase-2
    :meth:`cadtopo.agent.Agent.step` deliverable — the model's plain-text reply
    in full (there is no private channel; every deliverable is public).
    ``tool_calls`` is a deterministic log of every tool the agent executed
    during the pass (name/arguments/truncated result), captured harness-side —
    never parsed from the model's own text.
    """

    public: str = ""
    query: str = ""
    key: str = ""
    accuracy: float = 0.5
    raw: str = ""
    tool_calls: List[Dict[str, str]] = field(default_factory=list)
