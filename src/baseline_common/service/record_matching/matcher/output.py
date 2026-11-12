"""What a record-matching decision answers: the candidate it is, or why none is."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MatchVerdict(StrEnum):
    """The verdict on a record as a document wrote it."""

    MATCH = "match"
    """It is one of the candidates."""

    NEW_RECORD = "new_record"
    """It is none of them; a person creates it in the reference system."""

    UNCERTAIN = "uncertain"
    """The evidence does not settle it beyond reasonable doubt; a person decides."""


class MatchDecision(BaseModel):
    """The verdict, the candidate on a match, and what leaves it open when uncertain."""

    model_config = ConfigDict(frozen=True)

    verdict: MatchVerdict
    record_id: str | None = Field(
        None, description="With a match only: the id of the candidate it is."
    )
    reason: str | None = Field(
        None,
        min_length=1,
        description="With uncertain only, for the person deciding: what leaves it open, and "
        "between which candidates.",
    )
    leaning_id: str | None = Field(
        None,
        description="With uncertain only: the id of the candidate it leans toward, when one "
        "stands out; None when none does.",
    )

    @model_validator(mode="after")
    def check_fields_fit_the_verdict(self) -> Self:
        """Validate that each field comes with the verdict it belongs to.

        Returns:
            The decision, unchanged.

        Raises:
            ValueError: When an id, a reason or a leaning does not fit the verdict.
        """
        if (self.verdict is MatchVerdict.MATCH) != (self.record_id is not None):
            error_msg = "give the record_id with a match, and only with a match"
            raise ValueError(error_msg)

        uncertain = self.verdict is MatchVerdict.UNCERTAIN

        if uncertain != (self.reason is not None):
            error_msg = "give the reason with an uncertain verdict, and only with one"
            raise ValueError(error_msg)

        if self.leaning_id is not None and not uncertain:
            error_msg = "give the leaning_id with an uncertain verdict only"
            raise ValueError(error_msg)

        return self
