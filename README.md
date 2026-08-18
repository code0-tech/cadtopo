# CADTopo

Cost-aware, dynamic-topology multi-agent LLM orchestration. CADTopo implements
[DyTopo](https://arxiv.org/abs/2602.06039)'s protocol — per-round agent
selection, a communication topology induced from the agents' own
offer/demand descriptors, and a meta-agent that scores each round and
decides whether to halt — plus a cost-aware routing extension.

## Why

Static multi-agent pipelines (fixed roles, fixed order) waste calls on
agents that aren't relevant to a given task, and can't skip ahead when a
round already produced a correct answer. CADTopo instead:

1. picks which agents participate **each round**, based on how well their
   skill matches the current goal (a cheap embedding gate, no LLM call);
2. lets the participating agents run **in parallel**, each on its own view
   of the task and its private memory;
3. **induces a communication graph** after the fact, from what each agent
   says it can offer and what it still needs — so information routes to
   where it's useful without a fixed pipeline;
4. has a **manager** score the round's best output and either halt or set a
   focused goal for the next round, instead of running a fixed number of
   rounds regardless of progress.

## Architecture

```
src/cadtopo/
├── agent.py          SubModel — one worker agent (role, backbone, optional tools)
├── router.py          CADTopoRouter — coarse selection (Stage 1) + topology induction (Stage 2/3)
├── manager.py          ManagerModel — the meta-agent: scores each round, sets the next goal
├── orchestrator.py     CADTopoSystem — runs the full round loop end to end
├── parsing.py           Delimiter-based response parsing shared by agents + manager
├── llm.py                completion_with_retry — the one choke point for every LLM call
├── embedding.py           EmbeddingModel — local sentence-transformers backend for routing
├── telemetry.py           CostTracker + CallLog — token accounting and a verbatim call log
├── logging_utils.py        Coloured console logging
├── tracing.py               Recording wrappers + TraceCollector, for persisting a full run
└── benchmarks/
    └── humaneval.py          HumanEval loader + sandboxed evaluator, used by the example
```

**Round loop** (`CADTopoSystem.run`, see its docstring for the full detail):

1. **Coarse selection** — the router embeds the round's goal and keeps every
   agent whose static skill description clears a similarity threshold θ.
2. **Barrier** — every selected agent runs exactly one pass, in parallel,
   conditioned only on the task, its own memory, and the round goal. Each
   pass yields a public message (the contribution), a private message (a
   hand-off note for whoever ends up reading it), and a query/key
   descriptor pair (what it needs / what it offers).
3. **Topology induction** — the router embeds every agent's key/query text
   and scores each ordered pair `(i, j)` on offer→demand fit, `i`'s
   self-assessed accuracy, and `i`'s cost. Pairs above a threshold τ become
   directed edges; edges gate which private messages a later round's
   agents will see, not this round's execution.
4. **Manager review** — the manager reads the round's goal and every public
   message, scores the best candidate's completeness (Φ ∈ [0, 1]), and
   either halts (Φ ≥ γ) or drafts a focused goal for the next round.

The deliverable is always the highest-Φ round's best output — never a
separate synthesis pass.

## Install

```bash
pip install -e ".[dev]"
# or, with uv:
uv sync --extra dev
```

Requires Python ≥3.10. `sentence-transformers`/`torch` are used for the
local embedding backend; the LLM calls themselves go through
[LiteLLM](https://github.com/BerriAI/litellm), so any LiteLLM-supported
provider works.

## Quick start

```python
from cadtopo import CADTopoSystem, CADTopoRouter, ManagerModel, SubModel, EmbeddingModel

agents = [
    SubModel(
        name="Developer",
        skill_definition="Implement complete, runnable code.",
        system_prompt="You are Developer, the Code Implementation Specialist. ...",
        cost_per_token=1.0,
        api_provider="openrouter/meta-llama/llama-3.1-8b-instruct",
        api_key="sk-...",
    ),
    # ... more roles
]

router = CADTopoRouter(agents=agents, embedding_model=EmbeddingModel(), theta=0.2, tau=0.3)
manager = ManagerModel(api_provider="openrouter/meta-llama/llama-3.1-8b-instruct", api_key="sk-...")
system = CADTopoSystem(manager=manager, router=router, weights=(0.8, 0.1, 0.1), max_rounds=10)

answer = system.run("Implement a function that reverses a string.")
```

Every `SubModel.step` response must follow a small delimiter protocol
(`QUERY:`/`KEY:`/`ACCURACY:` header lines, then `===PRIVATE===`/`===PUBLIC===`
sections) — see `SubModel._STEP_FORMAT` and `cadtopo.parsing.parse_agent_step`.
Bake the equivalent instructions into each agent's `system_prompt`, or reuse
the ones in `examples/run_humaneval.py`, which appends them automatically.

### Example: HumanEval

`examples/run_humaneval.py` wires up DyTopo's four code-generation roles
(Developer, Researcher, Tester, Designer) and evaluates the system against a
random HumanEval sample, with full run tracing:

```bash
export PROVIDER=openrouter/meta-llama/llama-3.1-8b-instruct
export AUTH=sk-...
python examples/run_humaneval.py
```

Traces (every LLM call, every round's induced topology, the final answer,
and pass/fail) are written to `runs/<timestamp>/traces.jsonl` (machine
readable) and `runs/<timestamp>/traces.txt` (human readable).

## Testing

```bash
pytest
# with coverage:
pytest --cov=cadtopo
```

All LLM calls and the embedding backend are mocked or faked in tests — the
suite runs offline, with no network access and no real model calls. The one
exception is `tests/test_humaneval.py`'s evaluator tests, which execute
generated Python in a real subprocess (by design — that's what the
evaluator does), but only against fixed, trusted snippets.

## Design notes

- **Halting is a threshold, not a self-report.** Early experiments had the
  manager assert a yes/no "is this done?" bit directly, which made weaker
  backbones loop forever — there was always some plausible reason to answer
  "not yet". Scoring completeness as Φ ∈ [0, 1] and deriving the halting bit
  from `Φ ≥ γ_success` fixed this.
- **The manager doesn't choose the deliverable.** The round's best output is
  fixed mechanically as the coarse-selected agent with the highest skill
  match to the round's goal; the manager only scores that fixed pick and
  proposes the next goal. This keeps its job to *judging*, not *picking*,
  which measurably improved reliability on weaker backbones.
- **Everything parses as delimited text, not JSON.** Weak backbones reliably
  produce `LABEL: value` lines and `===MARKER===` sections, but reliably
  fail to JSON-escape a field once it contains code, quotes, or newlines.
  `cadtopo.parsing` standardises on the delimiter protocol everywhere
  (worker steps and manager reviews alike) instead.

## Scope

This repository holds the orchestration framework and one worked example
(HumanEval). It intentionally leaves out one-off experiment scripts from the
original research repo (a static-pipeline baseline for comparison, a
LiveBench harness, a trace-analysis notebook) — those depend on this
framework's public API and can be ported the same way if needed.
