"""Run CADTopo over HumanEval tasks via LiteLLM.

Provider and model are read from the environment::

    export PROVIDER=openrouter/meta-llama/llama-3.1-8b-instruct
    export AUTH=sk-...
    # optional:
    export API_BASE=https://...
    export HUMANEVAL_SEED=42
    # optional — sampling knobs for the manager's control call only
    # (see Manager.__init__): defaults to temperature=0.0, no
    # reasoning/thinking.
    export CONTROL_TEMPERATURE=0.0
    export CONTROL_REASONING_EFFORT=high      # OpenAI o-series / GPT-5
    export CONTROL_THINKING_BUDGET_TOKENS=2048  # Anthropic extended thinking

The whole team (the four roles, the router and the manager) is assembled by
:func:`agents.agents.build_system`; this example only loads the HumanEval
problems, feeds each one through the system, and grades the answer. The roles
and their prompts follow DyTopo's own code-generation setup (paper Appendix
B.1.1, Table 4).
"""

import ast
import gzip
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from agents.agents import API_BASE, AUTH, MAX_ROUNDS, MODEL_COSTS, PROVIDER, TAU, THETA, build_system
from cadtopo import llm
from cadtopo.logging_utils import get_logger
from cadtopo.telemetry import COST

_log = get_logger("cadtopo.examples.humaneval")

# Number of randomly selected HumanEval tasks to evaluate.
NUM_TASKS = 25
SEED = os.environ.get("HUMANEVAL_SEED")
# When set (BASELINE=1), bypass CADTopo entirely: send the raw prompt to the
# model in ONE plain call and grade that. Gives the single-pass number to
# compare the whole multi-agent pipeline against on the same tasks.
BASELINE = os.environ.get("BASELINE")


def baseline_answer(problem: Dict) -> str:
    """One direct model call, no CADTopo — the single-pass reference."""
    user = (
        "Implement the following Python function so that it passes all unit "
        "tests. Reply with ONLY the complete function (signature + any imports) "
        "as a single fenced ```python block, nothing else.\n\n"
        f"{problem['prompt']}"
    )
    return llm.text_completion(
        messages=[{"role": "user", "content": user}],
        model=PROVIDER, api_key=AUTH, api_base=API_BASE, label="Baseline",
    )

# The bundled dataset lives next to this example (``data/``); re-download there
# on first use.
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
_DATA_FILE = Path(__file__).resolve().parent / "data" / "HumanEval.jsonl.gz"


# --------------------------------------------------------------------------- #
# HumanEval loader & evaluator (stdlib only). Evaluating HumanEval means
# executing model-generated code; it runs in a separate subprocess with a
# timeout, but it is untrusted code by nature.
# --------------------------------------------------------------------------- #
def load_problems() -> List[Dict]:
    """Load all HumanEval problems, downloading & caching the dataset on first use.

    Each problem is a dict with keys: task_id, prompt, entry_point,
    canonical_solution, test.
    """
    if not _DATA_FILE.exists():
        _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        _log.info(f"Downloading HumanEval dataset to {_DATA_FILE}")
        import urllib.request

        urllib.request.urlretrieve(HUMANEVAL_URL, _DATA_FILE)

    problems: List[Dict] = []
    with gzip.open(_DATA_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def extract_code(answer: str, entry_point: str) -> str:
    """Extract the runnable Python solution from a (possibly prose-wrapped) model answer.

    Prefers a fenced ```python block that defines the target function;
    otherwise falls back to the longest fenced block, then to the raw
    answer. The result is mechanically sanitised (test scaffolding stripped,
    class wrappers unwrapped).
    """
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, re.DOTALL)

    with_entry = [b for b in blocks if f"def {entry_point}" in b]
    if with_entry:
        code = with_entry[-1].strip()
    elif blocks:
        code = max(blocks, key=len).strip()
    else:
        code = answer.strip()
    return _sanitize_code(code, entry_point)


class _SelfStripper(ast.NodeTransformer):
    """Rewrites ``self.x`` to ``x`` when hoisting a method to module level."""

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def _hoist_method(fn: ast.FunctionDef) -> ast.FunctionDef:
    """Turn a method into a module-level function (drops self/cls and decorators)."""
    if fn.args.args and fn.args.args[0].arg in ("self", "cls"):
        fn.args.args = fn.args.args[1:]
    fn.decorator_list = []
    return _SelfStripper().visit(fn)


def _is_main_guard(node: ast.stmt) -> bool:
    """Detects ``if __name__ == '__main__':`` blocks."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def _is_test_class(node: ast.stmt) -> bool:
    if not isinstance(node, ast.ClassDef):
        return False
    if any("TestCase" in ast.unparse(base) for base in node.bases):
        return True
    return node.name.startswith("Test")


def _sanitize_code(code: str, entry_point: str) -> str:
    """Strip test scaffolding from a solution and unwrap class-wrapped entry points.

    Model answers sometimes append unittest suites, ``unittest.main()``
    calls, or wrap the requested function in a class (LeetCode style). All of
    that breaks the canonical check, so it is removed mechanically:

      * unittest/doctest imports and TestCase-style classes are dropped,
      * ``if __name__ == '__main__'`` blocks and top-level call statements
        (test invocations, prints) are dropped,
      * if the entry point only exists as a method, its class's methods are
        hoisted to module level (dropping ``self``).

    Returns the code unchanged when it cannot be parsed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    has_top_level_entry = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point for n in tree.body
    )

    kept: List[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            node.names = [a for a in node.names if a.name not in ("unittest", "doctest")]
            if not node.names:
                continue
        if isinstance(node, ast.ImportFrom) and node.module in ("unittest", "doctest"):
            continue
        if _is_test_class(node) or _is_main_guard(node):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            continue
        if (
            not has_top_level_entry
            and isinstance(node, ast.ClassDef)
            and any(isinstance(m, ast.FunctionDef) and m.name == entry_point for m in node.body)
        ):
            kept.extend(_hoist_method(m) for m in node.body if isinstance(m, ast.FunctionDef))
            continue
        kept.append(node)

    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in kept):
        return code
    try:
        return ast.unparse(ast.Module(body=kept, type_ignores=[]))
    except Exception:  # noqa: BLE001 — fall back to the unsanitised code
        return code


def _build_program(problem: Dict, code: str) -> str:
    """Assemble a self-contained program: import preamble + solution + canonical test."""
    preamble = "\n".join(
        ln for ln in problem["prompt"].splitlines() if ln.startswith("import ") or ln.startswith("from ")
    )
    return f"{preamble}\n\n{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"


def evaluate(problem: Dict, answer: str, timeout: float = 15.0) -> Dict:
    """Run the canonical HumanEval test for one problem against a model answer.

    :return: ``{"passed": bool, "error": str, "code": str}``. The solution
        runs in a separate subprocess with a timeout.
    """
    entry_point = problem["entry_point"]
    code = extract_code(answer, entry_point)
    program = _build_program(problem, code)

    try:
        proc = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=timeout)
        passed = proc.returncode == 0
        error = "" if passed else (proc.stderr.strip()[-600:] or proc.stdout.strip()[-600:])
        return {"passed": passed, "error": error, "code": code}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Timeout after {timeout}s", "code": code}
    except Exception as e:  # noqa: BLE001 — report any harness failure as a non-pass
        return {"passed": False, "error": f"Harness error: {e}", "code": code}


def main():
    if not PROVIDER:
        raise SystemExit(
            "PROVIDER environment variable is required (e.g. 'openrouter/meta-llama/llama-3.1-8b-instruct').")

    system = None if BASELINE else build_system()

    problems = load_problems()
    if SEED is not None:
        random.seed(int(SEED))
    sample = random.sample(problems, min(NUM_TASKS, len(problems)))
    if BASELINE:
        _log.info(f"Evaluating {len(sample)}/{len(problems)} HumanEval tasks — SINGLE-PASS BASELINE (no CADTopo), {PROVIDER}")
    else:
        _log.info(
            f"Evaluating {len(sample)}/{len(problems)} HumanEval tasks with {PROVIDER} "
            f"(θ={THETA}, τ={TAU}, max_rounds={MAX_ROUNDS})")

    results: List[tuple] = []
    for idx, problem in enumerate(sample, start=1):
        task_id = problem["task_id"]
        _log.info(f"── TASK {idx}/{len(sample)} ── {task_id} (entry={problem['entry_point']})")

        cost_before = COST.snapshot()
        user_prompt = (
            "Implement the following Python function so that it passes all unit tests. "
            "Return the complete function including its signature and any required imports.\n\n"
            f"{problem['prompt']}"
        )

        final_answer = baseline_answer(problem) if BASELINE else system.run(user_prompt)
        outcome = evaluate(problem, final_answer)

        task_delta = COST.diff_since(cost_before)
        task_agg = COST.aggregate(task_delta)

        results.append((task_id, outcome["passed"], task_agg.calls, task_agg.total_tokens))
        if outcome["passed"]:
            _log.success(f"{task_id}: PASS  calls={task_agg.calls}  tokens={task_agg.total_tokens:,}")
        else:
            _log.error(f"{task_id}: FAIL  calls={task_agg.calls}  tokens={task_agg.total_tokens:,}")
            if outcome["error"]:
                _log.warning(f"  reason: {outcome['error'][:200]}")
            # Diagnostic: show what was actually evaluated, so a failure can be
            # classified as wrong-logic vs mangled-deliverable/extraction. The
            # raw answer length vs the extracted code reveals prose that never
            # made it into a code fence.
            raw = (final_answer or "").strip()
            fenced = "```" in raw
            _log.warning(f"  deliverable: {len(raw)} chars, fenced={fenced}; extracted code:\n{outcome['code'][:600]}")

    _print_summary(results)


def _print_summary(results: List[tuple]) -> None:
    passed = sum(1 for _, ok, _, _ in results if ok)
    total = len(results)
    rate = (passed / total * 100) if total else 0.0
    avg_tokens_per_task = (COST.total.total_tokens / total) if total else 0

    _log.info("─" * 60)
    _log.info("HUMANEVAL FINAL")
    _log.info("─" * 60)
    for task_id, ok, calls, tokens in results:
        line = f"  {'PASS' if ok else 'FAIL':<4} {task_id:<18} {calls:>3} calls  {tokens:>9,} tokens"
        (_log.success if ok else _log.error)(line)
    _log.info("─" * 60)
    (_log.success if passed == total else _log.info)(f"pass@1: {passed}/{total} ({rate:.1f}%)")
    _log.info(
        f"total: {COST.total.calls} LLM calls, {COST.total.total_tokens:,} tokens "
        f"(prompt={COST.total.prompt_tokens:,}, completion={COST.total.completion_tokens:,})"
    )
    _log.info(f"avg per task: {avg_tokens_per_task:,.0f} tokens")
    _log.info("per-agent breakdown:")
    for name, ac in sorted(COST.by_agent.items(), key=lambda kv: -kv[1].total_tokens):
        _log.info(
            f"  {name:<14} {ac.calls:>4} calls  {ac.total_tokens:>9,} tokens "
            f"(prompt={ac.prompt_tokens:,}, completion={ac.completion_tokens:,})")

    _log.info("per-model breakdown (cost = total tokens × $/1M):")
    total_cost = 0.0
    for model, ac in sorted(COST.by_model.items(), key=lambda kv: -kv[1].total_tokens):
        rate = MODEL_COSTS.get(model)
        if rate is not None:
            dollars = ac.total_tokens * rate / 1_000_000
            total_cost += dollars
            cost_str = f"${dollars:>8.4f} @ ${rate}/1M"
        else:
            cost_str = "     (no cost configured)"
        _log.info(
            f"  {model:<45} {ac.calls:>4} calls  {ac.total_tokens:>9,} tokens  {cost_str}")
    _log.info(f"estimated total cost: ${total_cost:.4f}")
    _log.info("─" * 60)


if __name__ == "__main__":
    main()
