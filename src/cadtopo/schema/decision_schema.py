from typing import Any

from pydantic import BaseModel, Field, field_validator


class RoundDecision(BaseModel):
    """The manager's per-round control output (DyTopo Eq. 13/14).

    The model reports the SAME judgement from two opposite framings in one
    call: ``failure_score`` = P(wrong) and ``success_score`` = P(right). Φ is
    the average of the two agreeing estimates,
    ``((1 - failure_score) + success_score) / 2`` (see
    :meth:`cadtopo.manager.Manager.review_round`) — a cheap self-consistency
    check that cancels one-sided over/under-confidence: a weak backbone that
    reflexively says "looks great" (low failure) but, asked directly for
    P(right), hesitates (mid success), lands in between instead of a
    false-positive halt. ``is_complete`` is NOT asked of the model — the
    orchestrator derives it by thresholding Φ against γ_success.

    Field ORDER is load-bearing (Hebel 1 — reason-before-answer): ``bug_hunt``
    is declared BEFORE ``failure_score`` so the structured-output backbone
    generates the worked trace FIRST and the score becomes its consequence.
    Weak backbones asked for the number up front emit ~58 tokens and reflex to
    the 0.5 midpoint; forcing the trace ahead of the score is what breaks that
    hedge.
    """

    best_agent: str = Field(
        description="The exact name of the agent whose output IS the deliverable (the finished artifact answering the request, never an evaluator's verdict)."
    )
    bug_hunt: str = Field(
        description="DO THE WORK HERE, BEFORE scoring. Take the hardest input among the examples STATED in the request, trace the deliverable on it step by step, and write its COMPUTED output next to the EXPECTED one. Then either name the exact input that makes it produce the wrong result, or state that you traced every stated example and none broke. This reasoning must precede — and justify — failure_score."
    )
    failure_score: float = Field(
        description="P(wrong) ∈ [0,1], justified by bug_hunt above: your estimated probability the chosen deliverable is DEFECTIVE — i.e. it would produce an incorrect result on at least one input the request cares about. If bug_hunt named a breaking input, this is near 1; if bug_hunt traced every stated example with no mismatch, near 0. Do not confirm success without a trace."
    )
    success_score: float = Field(
        description="P(right) ∈ [0,1], justified by the SAME bug_hunt above: your estimated probability the chosen deliverable is CORRECT — i.e. it produces the expected result on every input the request cares about. This is the opposite framing of failure_score; answer it independently (do not just compute 1 - failure_score). If bug_hunt traced every stated example with no mismatch, near 1; if it named a breaking input, near 0."
    )
    next_goal: str = Field(
        "", description="One short imperative sentence for the next round: the concrete thing to fix or add. Do not restate the task."
    )

    @field_validator("failure_score", "success_score", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0