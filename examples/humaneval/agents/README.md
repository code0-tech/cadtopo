# Agents

The four workers shared by the Inspect examples (`humaneval.py`,
`terminal_bench.py`), following DyTopo's setup (paper Appendix B.1.1, Table 4).
Each agent has its own folder holding two Markdown files; edit these to change an
agent's behaviour — no Python change needed.

`agents.py` factors out the system wiring: the prompt loaders, the shared
hyperparameters, `RoleSpec`, and `build_system()` (agents + router + manager →
`CadTopoAI`). An example only supplies its `RoleSpec`s and the tools each role
carries. `terminal_bench.py` reuses these same four roles unchanged — the
Developer/Tester just carry shell tools (`check_bash_syntax` / `run_bash`)
instead of the Python ones, so the prompt text still reads code-first.

| Folder | Role |
| --- | --- |
| `researcher/` | Algorithm Analyst — names the approach + complexity, no code. |
| `designer/` | API Signature Specialist — signatures + type hints only. |
| `developer/` | Code Implementation Specialist — writes the solution (uses `check_python_syntax`). |
| `tester/` | QA Engineer — executes the candidate (`run_python`) and emits a verdict. |

Each folder contains:

- `skill.md` — the router Stage-1 **skill descriptor** `S_i` (what the agent is
  for; used for coarse selection).
- `system.md` — the **role system prompt** sent on every call.

Tools (`check_python_syntax`/`run_python`, or `check_bash_syntax`/`run_bash`)
carry executable behaviour, so they stay in the example that uses them; only the
declarative prompt text lives here.
