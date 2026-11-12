"""What the boundary agent emits: one decision on one boundary, and its reason."""

from enum import StrEnum

from pydantic import BaseModel, Field


class BoundaryDecision(StrEnum):
    """The only two outcomes: the two sections are one (merge), or they are two (separate)."""

    MERGE = "merge"
    SEPARATE = "separate"


class BoundaryOpinion(BaseModel):
    """Whether the two sections are one section."""

    decision: BoundaryDecision
    reason: str = Field(description="One sentence, citing what in the text decided it.")
