"""Tests for the per-component, log-cost + budget cost-aware selector."""

import pytest

from cadtopo.agent import Agent
from cadtopo.backbone import Backbone
from cadtopo.manager import Manager
from cadtopo.selection import CostAwareSelector


def _agent(name="Developer", costs=(1.0, 3.0)) -> Agent:
    backbones = [Backbone(model=f"{name}-m{c}", cost=c) for c in reversed(costs)]
    return Agent(name=name, skill_definition="code", backbones=backbones)


def _manager(costs=(1.0, 3.0), gamma=0.8) -> Manager:
    backbones = [Backbone(model=f"mgr-m{c}", cost=c) for c in reversed(costs)]
    return Manager(backbones=backbones, gamma_success=gamma)


class TestLadderConstruction:
    def test_sorted_cheapest_first(self):
        agent = _agent(costs=(50.0, 0.1, 1.0))
        assert [b.cost for b in agent.backbones] == [0.1, 1.0, 50.0]
        assert agent.cost == 0.1

    def test_single_rung_is_a_no_op(self):
        agent = Agent(name="A", skill_definition="s", api_provider="m", cost_per_token=1.0)
        sel = CostAwareSelector()
        assert sel.for_work(agent, confidence=0.0, round_num=1, max_rounds=5).cost == 1.0

    def test_requires_a_backbone(self):
        with pytest.raises(ValueError):
            Agent(name="A", skill_definition="s")


class TestConfidenceDrivesTheRung:
    def test_confident_stays_cheapest(self):
        sel = CostAwareSelector()
        assert sel.for_work(_agent(), confidence=1.0, round_num=1, max_rounds=5).cost == 1.0

    def test_low_confidence_escalates_the_SAME_round(self):
        # The whole point: conf=0.1 in round 1 gets the strong model immediately,
        # with no dependence on the manager.
        sel = CostAwareSelector()
        assert sel.for_work(_agent(costs=(1.0, 3.0)), confidence=0.1, round_num=1, max_rounds=5).cost == 3.0

    def test_moderate_confidence_stays_cheap_early(self):
        sel = CostAwareSelector()
        assert sel.for_work(_agent(costs=(1.0, 3.0)), confidence=0.5, round_num=1, max_rounds=5).cost == 1.0


class TestBudgetChangesTheThreshold:
    def test_fewer_rounds_escalate_the_same_confidence_sooner(self):
        sel = CostAwareSelector()
        agent = _agent(costs=(1.0, 3.0))
        # Same conf=0.3, round 1 — with 5 rounds it holds cheap, with 2 it climbs.
        assert sel.for_work(agent, confidence=0.3, round_num=1, max_rounds=5).cost == 1.0
        assert sel.for_work(agent, confidence=0.3, round_num=1, max_rounds=2).cost == 3.0


class TestBudgetForcesEscalationLate:
    def test_manager_lukewarm_phi_escalates_before_budget_ends(self):
        # Φ=0.5 is the dead zone: risk 0.375 sits below the 2-rung midpoint, so
        # early rounds hold cheap — but a persistent 0.5 must be forced up as the
        # rounds run out (time GAIN > 1), not churn forever on the cheap model.
        sel = CostAwareSelector()
        mgr = _manager(costs=(1.0, 3.0), gamma=0.8)
        sel.observe(0.5)
        assert sel.for_control(mgr, round_num=1, max_rounds=5).cost == 1.0  # early: hold
        assert sel.for_control(mgr, round_num=5, max_rounds=5).cost == 3.0  # late: forced up
        # And it respects the budget: with only 3 rounds the same 0.5 climbs sooner.
        assert sel.for_control(mgr, round_num=3, max_rounds=3).cost == 3.0

    def test_stalled_lukewarm_worker_also_climbs_late(self):
        sel = CostAwareSelector()
        agent = _agent(costs=(1.0, 3.0))
        assert sel.for_work(agent, confidence=0.5, round_num=1, max_rounds=5).cost == 1.0
        assert sel.for_work(agent, confidence=0.5, round_num=5, max_rounds=5).cost == 3.0


class TestNonLinearCostLadder:
    def test_low_confidence_reaches_mid_early_top_only_late(self):
        sel = CostAwareSelector()
        agent = _agent(costs=(0.1, 1.0, 50.0))  # exponentially spaced
        # conf 0.1 early → the MID model (1.0), not the 50x one.
        assert sel.for_work(agent, confidence=0.1, round_num=1, max_rounds=5).cost == 1.0
        # same conf on the final round → the expensive top (50) is justified.
        assert sel.for_work(agent, confidence=0.1, round_num=5, max_rounds=5).cost == 50.0


class TestManagerUsesPreviousPhi:
    def test_round_one_no_phi_is_cheapest(self):
        sel = CostAwareSelector()
        mgr = _manager(costs=(1.0, 3.0))
        assert sel.for_control(mgr, round_num=1, max_rounds=5).cost == 1.0

    def test_low_previous_phi_escalates_the_manager(self):
        sel = CostAwareSelector()
        mgr = _manager(costs=(1.0, 3.0), gamma=0.8)
        sel.observe(0.0)  # last round was a total failure
        assert sel.for_control(mgr, round_num=2, max_rounds=5).cost == 3.0

    def test_recovered_phi_de_escalates(self):
        sel = CostAwareSelector()
        mgr = _manager(costs=(1.0, 3.0), gamma=0.8)
        sel.observe(0.0)
        assert sel.for_control(mgr, round_num=2, max_rounds=5).cost == 3.0
        sel.observe(0.9)  # Φ ≥ γ → risk 0
        assert sel.for_control(mgr, round_num=3, max_rounds=5).cost == 1.0


class TestPhaseOneAndReset:
    def test_describe_is_always_cheapest(self):
        sel = CostAwareSelector()
        assert sel.for_describe(_agent(costs=(0.1, 1.0, 50.0))).cost == 0.1

    def test_reset_clears_the_phi_signal(self):
        sel = CostAwareSelector()
        mgr = _manager(costs=(1.0, 3.0))
        sel.observe(0.0)
        sel.reset()
        assert sel.for_control(mgr, round_num=2, max_rounds=5).cost == 1.0
