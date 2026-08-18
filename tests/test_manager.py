import pytest

from cadtopo.manager import CompletedSubTask, Manager, aggregate_public
from cadtopo.schema import InitialGoal, RoundDecision


def _manager(**overrides) -> Manager:
    defaults = dict(api_provider="test-model", api_key="key", gamma_success=0.8)
    defaults.update(overrides)
    return Manager(**defaults)


class TestAggregatePublic:
    def test_concatenates_named_blocks_in_order(self):
        result = aggregate_public({"A": "first", "B": "second"})
        assert result == "--- [A] ---\nfirst\n\n--- [B] ---\nsecond"

    def test_empty_outputs_are_skipped(self):
        result = aggregate_public({"A": "", "B": "second"})
        assert result == "--- [B] ---\nsecond"

    def test_empty_dict_returns_empty_string(self):
        assert aggregate_public({}) == ""


class TestInitialGoal:
    def test_returns_the_model_goal(self, scripted_structured):
        scripted_structured([InitialGoal(goal="Produce a first draft solution.")])
        goal = _manager().initial_goal("Reverse a string.")
        assert goal == "Produce a first draft solution."

    def test_fails_open_to_generic_directive_on_error(self, scripted_structured):
        scripted_structured([ConnectionError("down")])
        goal = _manager().initial_goal("task")
        assert "first complete solution" in goal

    def test_blank_goal_fails_open(self, scripted_structured):
        scripted_structured([InitialGoal(goal="   ")])
        goal = _manager().initial_goal("task")
        assert "first complete solution" in goal


class TestReviewRound:
    def _review(self, scripted_structured, failure, next_goal="Fix the bug.", best="Developer", success=None, **kwargs):
        # Default success_score to the complement so Φ = ((1-f)+s)/2 collapses to
        # 1-failure, preserving the phi assertions written against the old formula.
        if success is None:
            success = 1.0 - failure
        scripted_structured([RoundDecision(best_agent=best, bug_hunt="traced", failure_score=failure, success_score=success, next_goal=next_goal)])
        manager = _manager(**kwargs)
        return manager.review_round(
            user_prompt="req", round_goal="goal", results={"Developer": "code"},
            completed=[], round_num=1, max_rounds=5,
        )

    def test_halts_when_phi_meets_gamma(self, scripted_structured):
        # failure_score=0.1 -> Φ = 1 - 0.1 = 0.9 ≥ γ=0.8, so it halts.
        decision = self._review(scripted_structured, failure=0.1, gamma_success=0.8)
        assert decision.is_complete is True
        assert decision.phi == 0.9
        assert decision.best_agent == "Developer"
        assert decision.best_result == "code"

    def test_manager_picks_the_named_deliverable_not_the_evaluator(self, scripted_structured):
        scripted_structured([RoundDecision(best_agent="Developer", bug_hunt="traced", failure_score=0.1, success_score=0.9, next_goal="Continue.")])
        decision = _manager().review_round(
            user_prompt="req", round_goal="goal",
            results={"Developer": "def f(): ...", "Tester": "VERDICT: CORRECT"},
            completed=[], round_num=1, max_rounds=5,
        )
        assert decision.best_agent == "Developer"
        assert decision.best_result == "def f(): ..."

    def test_unresolvable_pick_falls_back_to_first_output(self, scripted_structured):
        scripted_structured([RoundDecision(best_agent="Ghost", bug_hunt="traced", failure_score=0.1, success_score=0.9, next_goal="go")])
        decision = _manager().review_round(
            user_prompt="req", round_goal="g", results={"Developer": "code"},
            completed=[], round_num=1, max_rounds=5,
        )
        assert decision.best_agent == "Developer"
        assert decision.best_result == "code"

    def test_continues_with_next_goal_when_phi_below_gamma(self, scripted_structured):
        decision = self._review(scripted_structured, failure=0.7, next_goal="Handle empty input.", gamma_success=0.8)
        assert decision.is_complete is False
        assert decision.phi == pytest.approx(0.3)
        assert decision.next_goal == "Handle empty input."

    def test_transport_exception_fails_open_to_halt(self, scripted_structured):
        scripted_structured([ConnectionError("down")])
        decision = _manager().review_round(
            user_prompt="req", round_goal="", results={},
            completed=[], round_num=1, max_rounds=5,
        )
        assert decision.is_complete is True
        assert decision.phi == 1.0

    def test_continue_without_usable_next_goal_fails_open_to_halt(self, scripted_structured):
        # Below gamma, but the model left next_goal blank — must not loop forever.
        scripted_structured([RoundDecision(best_agent="A", bug_hunt="traced", failure_score=0.8, success_score=0.2, next_goal="")])
        decision = _manager(gamma_success=0.8).review_round(
            user_prompt="req", round_goal="", results={"A": "x"},
            completed=[], round_num=1, max_rounds=5,
        )
        assert decision.is_complete is True

    def test_verdict_agent_output_is_surfaced_as_primary_evidence(self, scripted_structured):
        scripted = scripted_structured([RoundDecision(best_agent="Developer", bug_hunt="traced", failure_score=0.8, success_score=0.2, next_goal="Fix the failing case.")])
        _manager(verdict_agent="Tester").review_round(
            user_prompt="req", round_goal="goal",
            results={"Developer": "code", "Tester": "VERDICT: INCORRECT\nf(2) returned 3, expected 4."},
            completed=[], round_num=1, max_rounds=5,
        )
        system = scripted.calls[0]["messages"][0]["content"]
        user = scripted.calls[0]["messages"][1]["content"]
        assert "TESTER VERDICT (from Tester" in user
        assert "f(2) returned 3, expected 4." in user
        assert "PRIMARY evidence" in system

    def test_no_verdict_output_degrades_to_plain_review(self, scripted_structured):
        scripted = scripted_structured([RoundDecision(best_agent="Developer", bug_hunt="traced", failure_score=0.5, success_score=0.5, next_goal="Continue.")])
        _manager(verdict_agent="Tester").review_round(
            user_prompt="req", round_goal="goal", results={"Developer": "code"},
            completed=[], round_num=1, max_rounds=5,
        )
        assert "TESTER VERDICT" not in scripted.calls[0]["messages"][1]["content"]
        assert "PRIMARY evidence" not in scripted.calls[0]["messages"][0]["content"]

    def test_control_temperature_is_forwarded(self, scripted_structured):
        scripted = scripted_structured([RoundDecision(best_agent="A", bug_hunt="traced", failure_score=0.1, success_score=0.9)])
        _manager(control_temperature=0.4).review_round(
            user_prompt="req", round_goal="", results={"A": "x"},
            completed=[], round_num=1, max_rounds=5,
        )
        assert scripted.calls[0]["temperature"] == 0.4

    def test_extra_control_params_are_forwarded(self, scripted_structured):
        scripted = scripted_structured([RoundDecision(best_agent="A", bug_hunt="traced", failure_score=0.1, success_score=0.9)])
        _manager(control_extra_params={"reasoning_effort": "high"}).review_round(
            user_prompt="req", round_goal="", results={"A": "x"},
            completed=[], round_num=1, max_rounds=5,
        )
        assert scripted.calls[0]["reasoning_effort"] == "high"


def test_completed_subtask_roundtrip():
    task = CompletedSubTask(goal="g", result="r")
    assert task.goal == "g"
    assert task.result == "r"
