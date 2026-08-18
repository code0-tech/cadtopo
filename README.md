<br/>
<br/>
<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/Codezero_Logo_White.svg">
  <img src="images/Codezero_Logo.svg" alt="CodeZero" width="200">
</picture>

### Every great idea starts at zero. <br/> Start with CodeZero.

**The open-source platform for building automations visually.<br/>Powered by AI, running wherever you want.**

[![Website](https://img.shields.io/badge/codezero.build-000000?logo=googlechrome&logoColor=white&style=for-the-badge)](https://codezero.build)
[![Docs](https://img.shields.io/badge/Docs-4B32C3?logo=readthedocs&logoColor=white&style=for-the-badge)](https://docs.codezero.build)
[![Discord](https://img.shields.io/discord/1173625923724124200?label=Discord&logo=discord&style=for-the-badge)](https://discord.com/invite/AyMB7DtA7P)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white&style=for-the-badge)](https://www.youtube.com/@CodeZeroBuild)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?logo=instagram&logoColor=white&style=for-the-badge)](https://www.instagram.com/codezero.tech/)

</div>

<br/>
<br/>

## What is CADTopo?

**CADTopo** (*Cost-Aware Dynamic-Topology*) is a framework for orchestrating **teams of LLM agents** that collaborate to solve a task. Instead of a fixed pipeline, the agents that participate — and how they pass information to each other — are decided **fresh every round**, and each agent runs on the **cheapest model that the current risk justifies**.

It is an implementation of the [DyTopo](https://arxiv.org/) protocol, extended with cost-aware model routing. In short:

- **Per-round agent selection** — only the agents relevant to the current goal are activated (no LLM calls wasted on the rest).
- **Dynamic topology** — the agents describe what they can *offer* and what they *need*, and the router builds the round's communication graph from those descriptors.
- **A manager (meta-agent)** — reads each round's output, picks the deliverable, scores it, and decides whether to stop or set a new goal.
- **Cost-aware routing** — every agent and the manager carry a *ladder* of models (cheap → expensive) and climb it only when the signal (low confidence, low score, running out of rounds) warrants it.

> **In one sentence:** the right agents, wired the right way, on the cheapest model that gets the job done — decided anew each round.

---

## How it works

Each **round** runs in two phases so information can flow *within* the round:

```
                    ┌─────────────────────────────────────────┐
   user task  ──▶   │ 1. COARSE SELECT   which agents run?     │  (embedding match, no LLM)
                    │ 2. DESCRIBE        what do they offer /   │  (cheap LLM pass)
                    │                    need this round?       │
                    │ 3. TOPOLOGY        build the round graph  │  (router, no LLM)
                    │ 4. WORK            run agents in order,    │  (the real work)
                    │                    routing hand-offs      │
                    │ 5. REVIEW          manager scores & picks │  (manager LLM pass)
                    └─────────────────────────────────────────┘
                                     │
                       halt?  ──▶ return best deliverable
                       else   ──▶ next round with a new goal
```

The manager halts as soon as the deliverable's score Φ crosses the success threshold; otherwise it keeps steering until the round cap is reached. The final answer is the highest-scoring round's deliverable.

### Core building blocks

| Component | What it does |
| --- | --- |
| `Agent` | A specialised worker (a role + a model ladder + optional tools). |
| `Router` | Coarse-selects agents and induces the per-round topology. |
| `Manager` | The meta-agent: scores each round, picks the deliverable, decides when to stop. |
| `CostAwareSelector` | Picks which model rung each component runs on this round. |
| `CadTopoAI` | The orchestrator that ties it all together and runs the rounds. |

---

## Installation

CADTopo targets **Python ≥ 3.10**. We recommend [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/code0-tech/cadtopo.git
cd cadtopo
uv sync            # install dependencies into a local .venv
```

Or with plain `pip`:

```bash
pip install -e .
```

Models are called through [LiteLLM](https://docs.litellm.ai/docs/providers), so you can use OpenAI, Anthropic, OpenRouter, local models, and more — just by changing the model string.

---

## Quickstart

```python
from cadtopo import Agent, Backbone, Router, Manager, EmbeddingModel, CadTopoAI

# 1. Define your agents (each with a role and a model).
agents = [
    Agent(
        name="Developer",
        skill_definition="Writes the Python implementation.",
        system_prompt="You are a senior Python developer. Return only the function.",
        api_provider="openrouter/meta-llama/llama-3.1-8b-instruct",
        api_key="sk-...",
    ),
    Agent(
        name="Tester",
        skill_definition="Reviews and validates the implementation.",
        system_prompt="You are a QA engineer. Point out any bugs.",
        api_provider="openrouter/meta-llama/llama-3.1-8b-instruct",
        api_key="sk-...",
    ),
]

# 2. Wire the router and manager.
router = Router(agents=agents, embedding_model=EmbeddingModel())
manager = Manager(api_provider="openrouter/meta-llama/llama-3.1-8b-instruct", api_key="sk-...")

# 3. Build the system and run it.
system = CadTopoAI(manager=manager, router=router, max_rounds=5)
answer = system.run("Implement a function that reverses a string.")
print(answer)
```

### Giving an agent a cost-aware model ladder

Pass multiple `Backbone`s instead of a single model, cheapest first. The selector climbs the ladder only when an agent is unsure or the round budget runs low:

```python
Agent(
    name="Developer",
    skill_definition="Writes the Python implementation.",
    backbones=[
        Backbone(model="openrouter/meta-llama/llama-3.1-8b-instruct", cost=0.06, api_key="sk-..."),
        Backbone(model="openrouter/anthropic/claude-3.5-sonnet",      cost=0.20, api_key="sk-..."),
        Backbone(model="openrouter/openai/gpt-5",                     cost=30.0, api_key="sk-..."),
    ],
)
```

Only the *relative* order of the `cost` values matters to the selector.

---

## Example: HumanEval

A complete, runnable example lives in [`examples/humaneval/`](examples/humaneval/). It runs a four-role team (Researcher, Designer, Developer, Tester) over the [HumanEval](https://github.com/openai/human-eval) coding benchmark and reports pass@1 plus a per-model cost breakdown.

```bash
cd examples/humaneval
cp .env.example .env          # then fill in PROVIDER and AUTH
uv run humaneval.py
```

The agents' roles and prompts are plain Markdown under `agents/<role>/` — edit them to change behaviour, no Python needed. Set `BASELINE=1` to bypass CADTopo and get a single-pass reference number on the same tasks.

---

## Project layout

```
src/cadtopo/
  orchestrator.py   # CadTopoAI — runs the rounds
  router.py         # coarse selection + topology induction
  manager.py        # the meta-agent (scoring, halting, next goal)
  agent.py          # the worker agent
  selection.py      # cost-aware model-ladder selection
  backbone.py       # a single model + its cost
  embedding.py      # skill/goal/query/key matching
  tools.py          # native tool-calling support
  schema/           # pydantic schemas for structured LLM output
examples/humaneval/ # end-to-end benchmark example
tests/              # test suite
```

Run the tests with:

```bash
uv run pytest
```

---

## License

Licensing varies per component. See the [LICENSE](LICENSE) file in this repository and in each subproject for details.

---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/CodeZero_Icon_White.svg">
  <img src="images/CodeZero_Icon.svg" alt="CodeZero icon" width="48">
</picture>

*Made with ❤️ by the CodeZero community*

</div>
