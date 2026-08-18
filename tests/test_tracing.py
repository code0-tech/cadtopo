import json

import numpy as np

from cadtopo.agent import Agent
from cadtopo.manager import CompletedSubTask, Manager
from cadtopo.router import Router
from cadtopo.schema import InitialGoal, RoundDecision
from cadtopo.tracing import RecordingManager, RecordingRouter, RecordingSubModel, TraceCollector, make_run_dir
from tests.conftest import FakeEmbedder, make_completion_response


class TestTraceCollectorLifecycle:
    def test_start_end_task_writes_jsonl_and_txt(self, tmp_path):
        trace = TraceCollector(tmp_path)
        trace.start_task(task_id="T1", user_prompt="do it", entry_point="f")
        trace.record_final_answer("the answer")
        trace.record_outcome(passed=True, error=None, calls=2, prompt_tokens=10, completion_tokens=5)
        trace.end_task()

        jsonl_lines = trace.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(jsonl_lines) == 1
        record = json.loads(jsonl_lines[0])
        assert record["task_id"] == "T1"
        assert record["final_answer"] == "the answer"
        assert record["outcome"] == {"passed": True, "error": None}
        assert record["cost"]["total_tokens"] == 15

        txt = trace.txt_path.read_text(encoding="utf-8")
        assert "TASK: T1" in txt
        assert "the answer" in txt

    def test_rerun_into_same_dir_truncates_rather_than_appends(self, tmp_path):
        trace1 = TraceCollector(tmp_path)
        trace1.start_task("T1", "p", "f")
        trace1.end_task()
        assert len(trace1.jsonl_path.read_text(encoding="utf-8").strip().splitlines()) == 1

        trace2 = TraceCollector(tmp_path)  # re-opening the same run_dir
        assert trace2.jsonl_path.read_text(encoding="utf-8") == ""

    def test_recording_methods_are_noops_without_an_active_task(self, tmp_path):
        trace = TraceCollector(tmp_path)
        # None of these should raise even though start_task was never called.
        trace.record_final_answer("x")
        trace.record_outcome(True, None, 1, 1, 1)
        trace.record_round_result("x")
        trace.record_review_decision(True, "")
        trace.record_routing("g", [], [[0]], [], {})
        trace.end_task()

    def test_multiple_tasks_append_separate_records(self, tmp_path):
        trace = TraceCollector(tmp_path)
        for i in range(3):
            trace.start_task(f"T{i}", "p", "f")
            trace.end_task()
        lines = trace.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(l)["task_id"] for l in lines] == ["T0", "T1", "T2"]


class TestRoutingAndStepRecording:
    def test_agent_steps_are_buffered_then_attached_at_routing(self, tmp_path):
        trace = TraceCollector(tmp_path)
        trace.start_task("T1", "p", "f")

        class FakeStep:
            query, key, accuracy, private, public, tool_calls = "q", "k", 0.7, "priv", "pub", []

        trace.record_agent_step("Developer", "memory", FakeStep())
        trace.record_routing(
            goal="g", active_agents=["Developer"], adjacency=[[0]], execution_order=["Developer"], profiles={},
        )
        trace.end_task()

        record = json.loads(trace.jsonl_path.read_text(encoding="utf-8").strip())
        round0 = record["rounds"][0]
        assert round0["agent_steps"][0]["agent"] == "Developer"
        assert round0["agent_steps"][0]["public"] == "pub"

    def test_completed_subtasks_backfill_round_results(self, tmp_path):
        trace = TraceCollector(tmp_path)
        trace.start_task("T1", "p", "f")
        trace.record_routing(goal="g", active_agents=[], adjacency=[], execution_order=[], profiles={})
        trace.record_completed_subtasks([CompletedSubTask(goal="g", result="the result")])
        trace.end_task()

        record = json.loads(trace.jsonl_path.read_text(encoding="utf-8").strip())
        assert record["rounds"][0]["round_result"] == "the result"


class TestRecordingSubModel:
    def test_step_is_recorded_and_delegates_to_super(self, tmp_path, scripted_llm):
        trace = TraceCollector(tmp_path)
        trace.start_task("T1", "p", "f")
        scripted_llm([make_completion_response("the output")])

        agent = RecordingSubModel(
            name="Developer", skill_definition="code", cost_per_token=1.0,
            api_provider="test-model", trace=trace,
        )
        step = agent.step("task", "", [])

        assert step.public == "the output"
        trace.record_routing(goal="", active_agents=["Developer"], adjacency=[[0]], execution_order=["Developer"], profiles={})
        trace.end_task()
        record = json.loads(trace.jsonl_path.read_text(encoding="utf-8").strip())
        assert record["rounds"][0]["agent_steps"][0]["public"] == "the output"


class TestRecordingRouter:
    def test_induce_topology_records_topology_without_embedding_vectors(self, tmp_path):
        trace = TraceCollector(tmp_path)
        trace.start_task("T1", "p", "f")

        embedder = FakeEmbedder({"": np.zeros(2)})
        agent = Agent(name="A", skill_definition="", cost_per_token=0.0, api_provider="test-model")
        router = RecordingRouter(agents=[agent], embedding_model=embedder, trace=trace)

        router.induce_topology([agent], {"A": {"key": "", "query": "", "accuracy": 0.5}})
        trace.end_task()

        record = json.loads(trace.jsonl_path.read_text(encoding="utf-8").strip())
        profile = record["rounds"][0]["profiles"]["A"]
        assert "key_emb" not in profile  # embedding vectors are stripped before recording
        assert profile["accuracy"] == 0.5


class TestRecordingManager:
    def test_review_round_records_the_control_call_and_decision(self, tmp_path, scripted_structured):
        trace = TraceCollector(tmp_path)
        trace.start_task("T1", "p", "f")
        trace.record_routing(goal="", active_agents=["A"], adjacency=[[0]], execution_order=["A"], profiles={})

        scripted_structured([RoundDecision(best_agent="A", bug_hunt="traced", failure_score=0.1, success_score=0.9, next_goal="go")])
        manager = RecordingManager(api_provider="test-model", api_key="k", trace=trace)

        decision = manager.review_round(
            user_prompt="req", round_goal="", results={"A": "x"},
            completed=[], round_num=1, max_rounds=5,
        )
        trace.end_task()

        assert decision.is_complete is True
        record = json.loads(trace.jsonl_path.read_text(encoding="utf-8").strip())
        round0 = record["rounds"][0]
        assert '"failure_score": 0.1' in round0["control_call"]["response"]
        assert round0["review"]["is_complete"] is True
        assert round0["review"]["phi"] == 0.9

    def test_initial_goal_is_recorded_at_task_level(self, tmp_path, scripted_structured):
        trace = TraceCollector(tmp_path)
        trace.start_task("T1", "p", "f")
        scripted_structured([InitialGoal(goal="Do the thing.")])
        manager = RecordingManager(api_provider="test-model", api_key="k", trace=trace)

        goal = manager.initial_goal("task")
        trace.end_task()

        assert goal == "Do the thing."
        record = json.loads(trace.jsonl_path.read_text(encoding="utf-8").strip())
        assert "Do the thing." in record["initial_goal_call"]["response"]


def test_make_run_dir_is_timestamped_under_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = make_run_dir(base="runs")
    assert run_dir.parent.name == "runs"
    assert run_dir.name  # non-empty timestamp component
