from typing import Any

from pydantic import BaseModel, Field, field_validator


class Descriptors(BaseModel):
    """Phase-1 routing descriptors (DyTopo Sec. 3.3).

    A lightweight, work-free declaration the router uses to induce the round's
    topology + aggregation order σ(t) before any agent does its work.
    """

    query: str = Field(
        description="One plain-English sentence: what you still need from other agents this round. Not code, not JSON."
    )
    key: str = Field(
        description="One plain-English sentence: what you can provide to other agents this round. Not code, not JSON."
    )
    failure_risk: float = Field(
        0.5,
        description=(
            "A single float 0.0-1.0: the probability you will FAIL to contribute "
            "correctly this round — P(wrong), NOT a confidence. Hunt for what could "
            "trip you up on THIS round's goal; assume the task is harder than it "
            "looks. Report near 0.0 only if you are genuinely certain you can nail "
            "it, near 1.0 if you likely cannot. Do NOT reflexively pick a low, "
            "self-flattering value."
        ),
    )

    @field_validator("failure_risk", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("query", "key")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        # A blank descriptor collapses routing to "no data exchange"; reject it
        # so Instructor re-asks for a real sentence.
        if not (v or "").strip():
            raise ValueError("must be a non-empty plain-English sentence")
        return v.strip()

