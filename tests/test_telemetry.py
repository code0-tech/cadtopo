from types import SimpleNamespace

from cadtopo.telemetry import AgentCost, CallLog, CostTracker


def usage(prompt=10, completion=5, total=None):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total or prompt + completion)


class TestCostTracker:
    def test_record_accumulates_per_agent_and_total(self):
        tracker = CostTracker()
        tracker.record("Developer", usage(10, 5))
        tracker.record("Developer", usage(20, 10))
        tracker.record("Tester", usage(1, 1))

        assert tracker.by_agent["Developer"] == AgentCost(calls=2, prompt_tokens=30, completion_tokens=15, total_tokens=45)
        assert tracker.total == AgentCost(calls=3, prompt_tokens=31, completion_tokens=16, total_tokens=47)

    def test_record_accumulates_per_model_when_given(self):
        tracker = CostTracker()
        tracker.record("Developer", usage(10, 5), model="cheap/model")
        tracker.record("Tester", usage(20, 10), model="cheap/model")
        tracker.record("Developer", usage(1, 1), model="strong/model")

        assert tracker.by_model["cheap/model"] == AgentCost(calls=2, prompt_tokens=30, completion_tokens=15, total_tokens=45)
        assert tracker.by_model["strong/model"] == AgentCost(calls=1, prompt_tokens=1, completion_tokens=1, total_tokens=2)
        # per-agent + total accounting is unaffected by the model split
        assert tracker.by_agent["Developer"].calls == 2
        assert tracker.total == AgentCost(calls=3, prompt_tokens=31, completion_tokens=16, total_tokens=47)

    def test_record_without_model_leaves_by_model_empty(self):
        tracker = CostTracker()
        tracker.record("Developer", usage(10, 5))
        assert tracker.by_model == {}

    def test_none_usage_is_ignored(self):
        tracker = CostTracker()
        tracker.record("Developer", None)
        assert tracker.total.calls == 0
        assert "Developer" not in tracker.by_agent

    def test_missing_total_tokens_field_is_derived(self):
        tracker = CostTracker()
        usage_no_total = SimpleNamespace(prompt_tokens=3, completion_tokens=4)
        tracker.record("Developer", usage_no_total)
        assert tracker.by_agent["Developer"].total_tokens == 7

    def test_snapshot_and_diff_since(self):
        tracker = CostTracker()
        tracker.record("Developer", usage(10, 5))
        baseline = tracker.snapshot()

        tracker.record("Developer", usage(20, 10))
        tracker.record("Tester", usage(1, 1))

        delta = tracker.diff_since(baseline)
        assert delta.keys() == {"Developer", "Tester"}
        assert delta["Developer"] == AgentCost(calls=1, prompt_tokens=20, completion_tokens=10, total_tokens=30)
        assert delta["Tester"] == AgentCost(calls=1, prompt_tokens=1, completion_tokens=1, total_tokens=2)

    def test_diff_since_omits_unchanged_agents(self):
        tracker = CostTracker()
        tracker.record("Developer", usage())
        baseline = tracker.snapshot()
        tracker.record("Tester", usage())
        delta = tracker.diff_since(baseline)
        assert "Developer" not in delta

    def test_snapshot_is_a_deep_copy(self):
        tracker = CostTracker()
        tracker.record("Developer", usage(10, 5))
        snap = tracker.snapshot()
        tracker.record("Developer", usage(10, 5))
        assert snap["Developer"].calls == 1  # unaffected by the later record

    def test_aggregate_sums_a_delta(self):
        delta = {
            "Developer": AgentCost(calls=1, prompt_tokens=10, completion_tokens=5, total_tokens=15),
            "Tester": AgentCost(calls=2, prompt_tokens=4, completion_tokens=4, total_tokens=8),
        }
        total = CostTracker.aggregate(delta)
        assert total == AgentCost(calls=3, prompt_tokens=14, completion_tokens=9, total_tokens=23)


class TestCallLog:
    def test_record_captures_label_model_messages_response_and_tokens(self):
        log = CallLog()
        log.set_label("Developer")
        log.record(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            response="hello",
            usage=usage(10, 5),
        )
        call = log.calls[0]
        assert call.seq == 1
        assert call.label == "Developer"
        assert call.model == "gpt-4o"
        assert call.messages == [{"role": "user", "content": "hi"}]
        assert call.response == "hello"
        assert call.total_tokens == 15

    def test_default_label_is_unknown_until_set(self):
        log = CallLog()
        log.record(model="m", messages=[], response="r", usage=None)
        assert log.calls[0].label == "unknown"

    def test_tool_call_fields_preserved(self):
        log = CallLog()
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function"}]},
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ]
        log.record(model="m", messages=messages, response="", usage=None)
        clean = log.calls[0].messages
        assert clean[0]["tool_calls"] == [{"id": "1", "type": "function"}]
        assert clean[1]["tool_call_id"] == "1"

    def test_non_dict_messages_are_skipped(self):
        log = CallLog()
        log.record(model="m", messages=["not a dict", {"role": "user", "content": "ok"}], response="", usage=None)
        assert len(log.calls[0].messages) == 1

    def test_mark_and_since_slice_only_later_calls(self):
        log = CallLog()
        log.record(model="m", messages=[], response="first", usage=None)
        mark = log.mark()
        log.record(model="m", messages=[], response="second", usage=None)
        log.record(model="m", messages=[], response="third", usage=None)

        since = log.since(mark)
        assert [c.response for c in since] == ["second", "third"]
