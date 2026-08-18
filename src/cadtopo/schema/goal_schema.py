from pydantic import BaseModel, Field


class InitialGoal(BaseModel):
    """The manager's first round goal C_task(0) (DyTopo Eq. 13).

    Formulated ONCE per task, before round 1, so the agents never receive the
    raw request as the goal. It only STEERS round 1 and must not restate the
    task's details, examples, or signature.
    """

    goal: str = Field(
        description="One short imperative sentence steering round 1. Do not restate the task."
    )
