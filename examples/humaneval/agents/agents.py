"""Shared CADTopo system builder for the Inspect AI examples.

Every example (`humaneval.py`, `terminal_bench.py`) runs the *same* four-role
team, wired the same way. This module factors that wiring out of the individual
examples:

* the prompt loaders (`skill`, `system_prompt`) that read the Markdown under
  ``agents/<role>/``,
* the shared CADTopo hyperparameters and backbone configuration, and
* :func:`build_system`, which assembles the agents, the router and the manager
  into a ready-to-run :class:`cadtopo.orchestrator.CadTopoAI`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Literal

from pydantic import Field

from cadtopo.agent import Agent, Backbone
from cadtopo.embedding import EmbeddingModel
from cadtopo.manager import Manager
from cadtopo.orchestrator import CadTopoAI
from cadtopo.router import Router
from cadtopo.schema import Tool

# --------------------------------------------------------------------------- #
# Backbone configuration — each worker gets a cost-aware ladder of two rungs:
# a cheap PROVIDER and a stronger PROVIDER_STRONG (falls back to PROVIDER when
# unset). The cost-aware selector (see cadtopo.selection) picks the rung per
# round. The manager's control call uses its own model. Read once from the env.
# --------------------------------------------------------------------------- #
PROVIDER = os.environ.get("PROVIDER")
AUTH = os.environ.get("AUTH")
API_BASE = os.environ.get("API_BASE")
# Stronger (more expensive) rung. Own auth/base optional; each falls back to the
# cheap rung's when unset, so a single-provider setup still works.
PROVIDER_STRONG = os.environ.get("PROVIDER_STRONG", PROVIDER)
AUTH_STRONG = os.environ.get("AUTH_STRONG", AUTH)
API_BASE_STRONG = os.environ.get("API_BASE_STRONG", API_BASE)

PROVIDER_VERY_STRONG = os.environ.get("PROVIDER_VERY_STRONG", PROVIDER)
AUTH_VERY_STRONG = os.environ.get("AUTH_VERY_STRONG", AUTH)
API_BASE_VERY_STRONG = os.environ.get("API_BASE_VERY_STRONG", API_BASE)

# Per-rung cost, in $ per 1M tokens where a token is prompt OR completion (the
# two are summed — one blended price for total tokens, not split in/out). Only
# the RELATIVE order matters to the selector; the absolute value is what the
# end-of-run cost report multiplies each model's token count by.
PROVIDER_COST = 0.06
PROVIDER_STRONG_COST = 0.20
PROVIDER_VERY_STRONG_COST = 30.0
# Model string → cost per 1M tokens, for the cost breakdown at the end of a run.
# Built cheap-first so that if a single provider serves both rungs
# (PROVIDER_STRONG unset), the stronger price wins for that one model.
MODEL_COSTS = {
    m: c
    for m, c in ((PROVIDER, PROVIDER_COST), (PROVIDER_STRONG, PROVIDER_STRONG_COST), (PROVIDER_VERY_STRONG, PROVIDER_VERY_STRONG_COST))
    if m
}

CONTROL_TEMPERATURE = os.environ.get("CONTROL_TEMPERATURE")
CONTROL_REASONING_EFFORT = os.environ.get("CONTROL_REASONING_EFFORT")
CONTROL_THINKING_BUDGET_TOKENS = os.environ.get("CONTROL_THINKING_BUDGET_TOKENS")

# --------------------------------------------------------------------------- #
# CADTopo hyperparameters (defaults; overridable per call to build_system).
# --------------------------------------------------------------------------- #
THETA = 0.4  # Gating threshold θ (router Stage 1: coarse selection).
# NOTE: skill embeddings for these roles score ~0.45–0.53 against a task, so a
# high θ (e.g. 0.8) gates out EVERYONE and the fallback keeps only the single
# best agent — the Tester never runs, so no code is ever executed and the
# Manager judges Φ blind. Keep θ low enough that the Tester participates.
TAU = 0.20  # Edge threshold τ (router Stage 2: topology)
MAX_ROUNDS = 1  # Hard round cap T_max; the manager usually halts earlier

# Prompts — loaded from Markdown in this directory (``agents/<role>/``).
AGENTS_DIR = Path(__file__).parent


def skill(role: str) -> str:
    """The router Stage-1 skill descriptor S_i for ``role`` (``<role>/skill.md``)."""
    return (AGENTS_DIR / role / "skill.md").read_text(encoding="utf-8").strip()


def system_prompt(role: str) -> str:
    """The role system prompt for ``role`` (``<role>/system.md``)."""
    return (AGENTS_DIR / role / "system.md").read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# Tools — executable behaviour the workers carry. The class docstring becomes
# the tool description the model sees (see :func:`cadtopo.tools.tool_schema`).
# --------------------------------------------------------------------------- #
class CheckPythonSyntax(Tool):
    """Check whether a Python source string is syntactically valid. Returns
    'OK: no syntax errors.' or the exact SyntaxError (message, line, column)."""

    action: Literal["check_python_syntax"] = "check_python_syntax"
    code: str = Field(description="The complete Python source code to check.")

    def run(self) -> str:
        """Compile-check ``code`` for syntax errors without executing it.

        The Developer's tool: cheap, local, deterministic — a plain
        ``compile()`` check, not an LLM judgement, so it never hallucinates a
        false OK/error.
        """
        try:
            compile(self.code, "<developer-draft>", "exec")
        except SyntaxError as e:
            text = e.text.rstrip() if e.text else ""
            return f"SYNTAX ERROR at line {e.lineno}, column {e.offset}: {e.msg}\n  {text}"
        except (ValueError, TypeError) as e:
            return f"ERROR: {e}"
        return "OK: no syntax errors."


class RunPython(Tool):
    """Execute a complete, self-contained Python script in a fresh subprocess and
    return its STDOUT, STDERR and exit code. To test a candidate function, submit
    ONE script containing the candidate code followed by checks of the stated
    examples (print actual vs expected, or use asserts)."""

    action: Literal["run_python"] = "run_python"
    code: str = Field(description="The complete Python script to execute.")

    def run(self) -> str:
        """Execute a complete Python script in a fresh subprocess; report the outcome.

        The Tester's tool: lets it RUN the candidate solution against the
        request's own stated examples instead of judging by reading. Isolated
        interpreter (``-I``), 10s wall-clock cap, stdout/stderr truncated — the
        result the model sees is exactly what the script really did.
        """
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", self.code],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            return "TIMEOUT: execution exceeded 10 seconds (infinite loop or blocking call?)."
        parts = []
        if proc.stdout:
            parts.append(f"STDOUT:\n{proc.stdout[-2000:]}")
        if proc.stderr:
            parts.append(f"STDERR:\n{proc.stderr[-2000:]}")
        parts.append(f"EXIT CODE: {proc.returncode}" + ("" if proc.returncode == 0 else " (non-zero: the run failed)"))
        return "\n".join(parts)


def _control_extra_params() -> dict:
    """Assemble the manager's optional control-call sampling knobs from the env."""
    params: dict = {}
    if CONTROL_REASONING_EFFORT:
        params["reasoning_effort"] = CONTROL_REASONING_EFFORT
    if CONTROL_THINKING_BUDGET_TOKENS:
        params["thinking"] = {"type": "enabled", "budget_tokens": int(CONTROL_THINKING_BUDGET_TOKENS)}
    return params


def build_system() -> CadTopoAI:
    """Assemble the shared four-role CADTopo system.

    Each worker is registered one by one as an :class:`~cadtopo.agent.Agent`,
    each carrying its OWN two-rung cost-aware backbone ladder (cheap ``PROVIDER``
    + stronger ``PROVIDER_STRONG``); the agents feed a
    :class:`~cadtopo.router.Router` (Stage-1 θ selection + Stage-2 τ topology) and
    a :class:`~cadtopo.manager.Manager` (the meta-agent Π_meta), all wrapped in a
    :class:`~cadtopo.orchestrator.CadTopoAI`.

    Backbone/auth come from the ``PROVIDER``/``PROVIDER_STRONG`` (and
    ``AUTH``/``API_BASE``) environment variables; the hyperparameters are the
    module constants.
    """
    agents: List[Agent] = []

    agents.append(
        Agent(
            name="Researcher",
            skill_definition=skill("researcher"),
            system_prompt=system_prompt("researcher"),
            mandatory=False,
            backbones=[
                Backbone(model=PROVIDER, cost=PROVIDER_COST, api_key=AUTH, api_url=API_BASE),
                #Backbone(model=PROVIDER_STRONG, cost=PROVIDER_STRONG_COST, api_key=AUTH_STRONG, api_url=API_BASE_STRONG),
                #Backbone(model=PROVIDER_VERY_STRONG, cost=PROVIDER_VERY_STRONG_COST, api_key=AUTH_VERY_STRONG, api_url=API_BASE_VERY_STRONG),
            ],
        )
    )

    agents.append(
        Agent(
            name="Designer",
            skill_definition=skill("designer"),
            system_prompt=system_prompt("designer"),
            mandatory=False,
            backbones=[
                Backbone(model=PROVIDER, cost=PROVIDER_COST, api_key=AUTH, api_url=API_BASE),
                #Backbone(model=PROVIDER_STRONG, cost=PROVIDER_STRONG_COST, api_key=AUTH_STRONG, api_url=API_BASE_STRONG),
                #Backbone(model=PROVIDER_VERY_STRONG, cost=PROVIDER_VERY_STRONG_COST, api_key=AUTH_VERY_STRONG, api_url=API_BASE_VERY_STRONG),
            ],
        )
    )

    agents.append(
        Agent(
            name="Developer",
            skill_definition=skill("developer"),
            system_prompt=system_prompt("developer"),
            # No check_python_syntax tool: on weak backbones it wrecks the pass —
            # the model double-escapes newlines inside the tool's JSON args (every
            # call fails on "\\n"), dumps the tool's own source into its answer,
            # and triggers Groq "failed to call a function" deaths. Plain code
            # generation (no tool) is far cleaner.
            mandatory=False,
            backbones=[
                Backbone(model=PROVIDER, cost=PROVIDER_COST, api_key=AUTH, api_url=API_BASE),
                #Backbone(model=PROVIDER_STRONG, cost=PROVIDER_STRONG_COST, api_key=AUTH_STRONG, api_url=API_BASE_STRONG),
                #Backbone(model=PROVIDER_VERY_STRONG, cost=PROVIDER_VERY_STRONG_COST, api_key=AUTH_VERY_STRONG, api_url=API_BASE_VERY_STRONG),
            ],
        )
    )

    agents.append(
        Agent(
            name="Tester",
            skill_definition=skill("tester"),
            system_prompt=system_prompt("tester"),
            tools=[RunPython],
            mandatory=False,
            backbones=[
                Backbone(model=PROVIDER, cost=PROVIDER_COST, api_key=AUTH, api_url=API_BASE),
                #Backbone(model=PROVIDER_STRONG, cost=PROVIDER_STRONG_COST, api_key=AUTH_STRONG, api_url=API_BASE_STRONG),
                #Backbone(model=PROVIDER_VERY_STRONG, cost=PROVIDER_VERY_STRONG_COST, api_key=AUTH_VERY_STRONG, api_url=API_BASE_VERY_STRONG),
            ],
        )
    )

    router = Router(
        agents=agents,
        embedding_model=EmbeddingModel(),
        theta=THETA,
        tau=TAU,
    )

    manager = Manager(
        backbones=[
            Backbone(model=PROVIDER, cost=PROVIDER_COST, api_key=AUTH, api_url=API_BASE),
            #Backbone(model=PROVIDER_STRONG, cost=PROVIDER_STRONG_COST, api_key=AUTH_STRONG, api_url=API_BASE_STRONG),
            #Backbone(model=PROVIDER_VERY_STRONG, cost=PROVIDER_VERY_STRONG_COST, api_key=AUTH_VERY_STRONG, api_url=API_BASE_VERY_STRONG),
        ],
        control_temperature=float(CONTROL_TEMPERATURE) if CONTROL_TEMPERATURE is not None else 0.0,
        control_extra_params=_control_extra_params(),
        # NOTE: verdict_agent="Tester" is intentionally OFF. It only helps if the
        # Tester actually EXECUTES the candidate (run_python) — on weak backbones
        # the Tester emits "VERDICT: CORRECT" without running anything, and
        # trusting that ungrounded verdict makes the Manager confidently HALT on
        # broken code. Re-enable only once the Tester reliably executes.
    )

    return CadTopoAI(manager=manager, router=router, max_rounds=MAX_ROUNDS)
