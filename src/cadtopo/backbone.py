"""Backbone: one candidate model (+ its cost) a component may run on.

Both the worker :class:`~cadtopo.agent.Agent` and the
:class:`~cadtopo.manager.Manager` carry a LADDER of these (cheap → expensive);
a :class:`~cadtopo.selection.ModelSelector` picks which rung each uses per round.
"""

from dataclasses import dataclass
from typing import List, Optional

from .logging_utils import get_logger

_log = get_logger("cadtopo.backbone")


@dataclass
class Backbone:
    """One candidate model a component may run on, with its cost.

    :param model: LiteLLM model/provider string (e.g. ``openai/gpt-4o-mini``).
    :param cost: Inference cost per 1k tokens. Only the RELATIVE ordering within
        a component's ladder matters for selection; the router normalises it into
        the routing score (see :func:`cadtopo.router.Router.induce_topology`).
    :param api_key: API key for this backbone. ``None`` falls back to LiteLLM's
        own environment resolution.
    :param api_url: Optional custom base URL for this backbone.
    """

    model: str
    cost: float = 1.0
    api_key: Optional[str] = None
    api_url: Optional[str] = None


def resolve_backbones(
    name: str,
    backbones: Optional[List[Backbone]],
    api_provider: Optional[str],
    cost_per_token: Optional[float],
    api_key: Optional[str],
    api_url: Optional[str],
) -> List[Backbone]:
    """Normalise a component's ladder: either an explicit list or the
    single-backbone shorthand, always returned non-empty and sorted
    cheapest-first."""
    if backbones:
        if api_provider is not None:
            _log.warning(f"{name}: both 'backbones' and 'api_provider' given — ignoring the shorthand.")
        return sorted(backbones, key=lambda b: b.cost)
    if api_provider is None:
        raise ValueError(f"{name}: provide either 'backbones' or 'api_provider'.")
    return [Backbone(model=api_provider, cost=cost_per_token if cost_per_token is not None else 1.0,
                     api_key=api_key, api_url=api_url)]
