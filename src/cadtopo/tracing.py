"""Recording wrappers + trace collector for observing a CADTopo run.

The wrapper classes subclass :class:`Manager`/:class:`Router`/
:class:`Agent` and forward every public call to ``super()`` after
capturing the inputs and outputs into a :class:`TraceCollector`. Per task, one
JSON object is appended to ``runs/<timestamp>/traces.jsonl`` (the
machine-readable record); the same record is also appended to
``runs/<timestamp>/traces.txt`` in a human-readable layout with real line
breaks instead of JSON-escaped ``\\n``.

Each JSON record contains:

  * ``task_id``, ``entry_point``, ``user_prompt``
  * ``rounds``: per round, the induced topology (active agents, adjacency, σ,
    profiles), every agent's single pass (memory in + query/key/accuracy +
    private/public), the round's picked best-agent result (or aggregated
    fallback), and the manager's review decision
  * ``llm_calls``: EVERY single LLM call made for this task, verbatim and in
    call order — captured at the choke point so nothing can be missed
  * ``final_answer``: the deliverable returned by the run
  * ``outcome``: passed + error
  * ``cost``: calls + prompt/completion/total tokens
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .agent import Agent
from .manager import CompletedSubTask, Manager, aggregate_public
from .router import Router
from .telemetry import CALLS

# Sentinel used when an agent has no prior context (first round). The txt
# renderer keys off this to omit the MEMORY IN block entirely.
_NO_MEMORY = "<no prior context — first round>"


class TraceCollector:
    """Accumulates per-task structured trace data and writes JSONL + txt.

    One ``start_task``/``end_task`` envelope per task. Recording methods are
    no-ops if no task is active, so a stray call never crashes a run.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = run_dir / "traces.jsonl"
        self.txt_path = run_dir / "traces.txt"
        self.jsonl_path.write_text("", encoding="utf-8")  # truncate so reruns into the same dir don't compound
        self.txt_path.write_text("", encoding="utf-8")
        self._current: Optional[dict] = None
        # The round's topology is induced in phase 1 (from the agents'
        # descriptors) BEFORE the phase-2 work passes run, so the round record
        # already exists by the time an agent step is recorded and steps
        # attach to it directly. This buffer is only a fallback for a step
        # recorded before any round record exists.
        self._pending_steps: list[dict] = []
        self._call_mark: int = 0

    # --- lifecycle ------------------------------------------------------ #
    def start_task(self, task_id: str, user_prompt: str, entry_point: str) -> None:
        self._current = {
            "task_id": task_id,
            "entry_point": entry_point,
            "user_prompt": user_prompt,
            "initial_goal_call": None,
            "rounds": [],
            "llm_calls": [],
            "final_answer": None,
            "outcome": None,
            "cost": None,
        }
        self._pending_steps = []
        self._call_mark = CALLS.mark()

    def end_task(self) -> None:
        if self._current is None:
            return
        self._current["llm_calls"] = [asdict(c) for c in CALLS.since(self._call_mark)]
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._current, ensure_ascii=False) + "\n")
        with self.txt_path.open("a", encoding="utf-8") as f:
            f.write(self._render_task(self._current))
        self._current = None

    # --- human-readable rendering ---------------------------------------- #
    @staticmethod
    def _render_task(task: dict) -> str:
        """Render one task record as plain text with real newlines."""
        width = 80
        rule = "=" * width
        sub = "-" * width

        def block(label: str, body: object) -> list[str]:
            return [f"{label}:", "" if body is None else str(body), ""]

        def framed(header: str, body: object) -> list[str]:
            """Render ``body`` inside a titled box so message boundaries stay
            unambiguous even when the content itself is multi-line."""
            text = "" if body is None else str(body)
            top = f"┌─ {header} "
            out = [top + "─" * max(0, width - len(top))]
            out += [f"│ {ln}" if ln else "│" for ln in text.split("\n")]
            out.append("└" + "─" * (width - 1))
            return out

        lines: list[str] = [rule, f"TASK: {task.get('task_id')}   entry_point: {task.get('entry_point')}", rule, ""]
        lines += block("USER PROMPT", task.get("user_prompt"))

        for i, r in enumerate(task.get("rounds") or [], 1):
            lines += [sub, f"ROUND {i}", sub, "", f"goal: {r.get('goal')}", f"active_agents: {r.get('active_agents')}"]
            lines += [f"execution_order (σ, drives phase-2 run order + memory): {r.get('execution_order')}", ""]

            for st in r.get("agent_steps") or []:
                lines += [
                    f">> AGENT: {st.get('agent')}  (accuracy={st.get('accuracy')})",
                    f"   QUERY: {st.get('query')}",
                    f"   KEY:   {st.get('key')}",
                    "",
                ]
                memory_in = st.get("memory_in") or ""
                if memory_in.strip() and not memory_in.startswith(_NO_MEMORY):
                    lines += block("   MEMORY IN", memory_in)
                tool_calls = st.get("tool_calls") or []
                if tool_calls:
                    lines += block(
                        "   TOOL CALLS (this pass)",
                        "\n".join(f"- {c.get('name')}({c.get('arguments')}) -> {c.get('result')}" for c in tool_calls),
                    )
                lines += block("   PRIVATE (routed to consumers)", st.get("private"))
                lines += block("   PUBLIC (to manager)", st.get("public"))

            review = r.get("review") or {}
            if review:
                lines += [
                    f"REVIEW: best_agent={review.get('best_agent')!r} phi={review.get('phi')} "
                    f"(success_score={review.get('success_score')}) is_complete={review.get('is_complete')} "
                    f"next_goal={review.get('next_goal')!r}",
                    "",
                ]
            lines += block("ROUND RESULT (best agent result, or aggregated fallback)", r.get("round_result"))

        lines += block("FINAL ANSWER (last global state S_global)", task.get("final_answer"))

        outcome = task.get("outcome") or {}
        cost = task.get("cost") or {}
        lines += [
            f"OUTCOME: {'PASS' if outcome.get('passed') else 'FAIL'}   error: {outcome.get('error')}",
            f"COST: {cost.get('calls')} calls, {cost.get('total_tokens')} tokens "
            f"(prompt={cost.get('prompt_tokens')}, completion={cost.get('completion_tokens')})",
            "",
        ]

        calls = task.get("llm_calls") or []
        lines += [rule, f"ALL LLM CALLS ({len(calls)})", rule, ""]
        for c in calls:
            lines += [
                sub,
                f"CALL #{c.get('seq')}  [{c.get('label')}]  model={c.get('model')}",
                f"tokens={c.get('total_tokens')} (prompt={c.get('prompt_tokens')}, completion={c.get('completion_tokens')})",
                sub, "", "▼ REQUEST (sent to LLM)", "",
            ]
            for m in c.get("messages") or []:
                role = str(m.get("role", "")).upper()
                body = m.get("content") or ""
                # Assistant tool-call REQUESTS carry no readable content — the
                # request lives in the separate tool_calls field.
                requested = m.get("tool_calls") or []
                if requested:
                    req_text = "\n".join(
                        f"→ TOOL CALL: {tc.get('function', {}).get('name')}({tc.get('function', {}).get('arguments')})"
                        for tc in requested
                    )
                    body = f"{body}\n{req_text}" if body else req_text
                if role == "TOOL":
                    tcid = m.get("tool_call_id")
                    role = f"TOOL RESULT (id={tcid})" if tcid else "TOOL RESULT"
                lines += framed(role, body)
                lines += [""]
            lines += ["▲ RESPONSE (from LLM)", ""]
            lines += framed("ASSISTANT", c.get("response"))
            lines += ["", ""]
        lines += [""]
        return "\n".join(lines)

    # --- recording API used by the wrapper classes ---------------------- #
    def record_manager_call(self, phase: str, system: str, user: str, response: str) -> None:
        if self._current is None:
            return
        call = {"system": system, "user": user, "response": response}
        if phase == "initial_goal":
            # Runs once per task, before round 1 — no round record exists yet.
            self._current["initial_goal_call"] = call
            return
        if not self._current["rounds"]:
            return
        if phase == "control":
            self._current["rounds"][-1]["control_call"] = call

    def record_review_decision(
        self, is_complete: bool, next_goal: str, best_agent: str = "", phi: float = 0.0, success_score: float = 0.0,
    ) -> None:
        if self._current is None or not self._current["rounds"]:
            return
        self._current["rounds"][-1]["review"] = {
            "is_complete": is_complete, "next_goal": next_goal,
            "best_agent": best_agent, "phi": phi, "success_score": success_score,
        }

    def record_round_result(self, result: str) -> None:
        if self._current is None or not self._current["rounds"]:
            return
        self._current["rounds"][-1]["round_result"] = result

    def record_round_goal(self, goal: str) -> None:
        """Backfill the current round's goal (known at manager review time)."""
        if self._current is None or not self._current["rounds"]:
            return
        if not self._current["rounds"][-1].get("goal"):
            self._current["rounds"][-1]["goal"] = goal

    def record_routing(
        self, goal: str, active_agents: list[str], adjacency: list[list[int]], execution_order: list[str], profiles: dict,
    ) -> None:
        if self._current is None:
            return
        agent_steps = self._pending_steps
        self._pending_steps = []
        self._current["rounds"].append({
            "goal": goal,
            "active_agents": active_agents,
            "adjacency": adjacency,
            "execution_order": execution_order,
            "profiles": profiles,
            "agent_steps": agent_steps,
            "round_result": None,
            "review": None,
            "control_call": None,
        })

    def record_agent_step(self, agent_name: str, memory_in: str, step: object) -> None:
        """Record one agent's phase-2 work output into the current round.

        The round record already exists (topology was induced in phase 1), so
        the step attaches straight to it; the buffer is a fallback only.
        """
        if self._current is None:
            return
        entry = {
            "agent": agent_name,
            "memory_in": memory_in,
            "query": getattr(step, "query", ""),
            "key": getattr(step, "key", ""),
            "accuracy": getattr(step, "accuracy", None),
            "private": getattr(step, "private", ""),
            "public": getattr(step, "public", ""),
            "tool_calls": list(getattr(step, "tool_calls", None) or []),
        }
        if self._current["rounds"]:
            self._current["rounds"][-1]["agent_steps"].append(entry)
        else:
            self._pending_steps.append(entry)

    def record_completed_subtasks(self, completed: list[CompletedSubTask]) -> None:
        """Backfill ``round_result`` for each round from the completed list."""
        if self._current is None:
            return
        for i, c in enumerate(completed):
            if i < len(self._current["rounds"]):
                self._current["rounds"][i]["round_result"] = c.result
            else:
                self._current["rounds"].append({
                    "goal": c.goal, "active_agents": [], "adjacency": [], "execution_order": [],
                    "profiles": {}, "agent_steps": [], "round_result": c.result, "review": None, "control_call": None,
                })

    def record_final_answer(self, answer: str) -> None:
        if self._current is None:
            return
        self._current["final_answer"] = answer

    def record_outcome(self, passed: bool, error: Optional[str], calls: int, prompt_tokens: int, completion_tokens: int) -> None:
        if self._current is None:
            return
        self._current["outcome"] = {"passed": passed, "error": error}
        self._current["cost"] = {
            "calls": calls, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


# --------------------------------------------------------------------------- #
# Recording wrappers
# --------------------------------------------------------------------------- #
class RecordingManager(Manager):
    """Wraps :class:`Manager` to record both per-round manager calls + the result."""

    def __init__(self, *args, trace: TraceCollector, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace = trace

    def _record(self, phase: str, system: str, user: str, obj) -> None:
        try:
            response = obj.model_dump_json(indent=2)
        except Exception:  # noqa: BLE001
            response = str(obj)
        self._trace.record_manager_call(phase, system, user, response)

    def review_round(
        self, user_prompt: str, round_goal: str, results, completed, round_num: int, max_rounds: int,
    ):
        # Backfill prior rounds' results so traces stay complete even when the
        # manager halts early.
        self._trace.record_completed_subtasks(completed)
        self._trace.record_round_goal(round_goal)
        decision = super().review_round(
            user_prompt, round_goal, results, completed, round_num, max_rounds,
        )
        # Mirrors what the orchestrator uses as its deliverable: the picked
        # best agent result, falling back to the aggregated public messages.
        self._trace.record_round_result(decision.best_result or aggregate_public(results or {}))
        self._trace.record_review_decision(
            decision.is_complete, decision.next_goal, decision.best_agent, decision.phi, decision.success_score,
        )
        return decision


class RecordingRouter(Router):
    """Wraps :class:`Router` to record the per-round topology it induces."""

    def __init__(self, *args, trace: TraceCollector, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace = trace

    def induce_topology(self, active_agents, profiles):
        decision = super().induce_topology(active_agents, profiles)
        # Profiles include embedding vectors; strip them — they bloat the
        # JSONL and are non-trivial to interpret post-hoc.
        slim_profiles = {
            name: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items() if not k.endswith("_emb")}
            for name, p in decision.profiles.items()
        }
        self._trace.record_routing(
            goal="",  # goal is recorded via the round context; kept for schema stability
            active_agents=[a.name for a in decision.active_agents],
            adjacency=decision.adjacency_matrix.astype(int).tolist(),
            execution_order=[decision.active_agents[i].name for i in decision.execution_order],
            profiles=slim_profiles,
        )
        return decision


class RecordingSubModel(Agent):
    """Wraps :class:`Agent` to record each agent's single-pass step."""

    def __init__(self, *args, trace: TraceCollector, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace = trace

    def step(self, task: str, round_goal: str, history=None):
        result = super().step(task, round_goal, history)
        self._trace.record_agent_step(self.name, self._render_memory_in(task, history), result)
        return result

    @staticmethod
    def _render_memory_in(task, history) -> str:
        """Flatten the pinned task + accumulated chat history into a readable
        block so the trace's MEMORY IN shows what the agent conditioned on
        besides this round's goal (the verbatim multi-turn request is also in
        ``llm_calls``)."""
        parts = [f"[TASK]\n{task}"] if task else []
        parts += [f"[{str(m.get('role', '')).upper()}]\n{m.get('content', '')}" for m in (history or [])]
        return "\n\n".join(parts) if parts else _NO_MEMORY


def make_run_dir(base: str = "runs") -> Path:
    """Create a timestamped run directory under ``base/``."""
    return Path(base) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
