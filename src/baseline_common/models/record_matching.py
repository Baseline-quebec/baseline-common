"""What record matching reads and answers: a written name, and which known organization it is."""

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from baseline_common.models.organization import Organization


class MatchRequest(BaseModel):
    """A name as a document wrote it, and the known organizations it may be."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    NAME_MAX_LENGTH: ClassVar[int] = 300
    """The longest name a document writes; anything longer is not a name."""
    CANDIDATES_MAX: ClassVar[int] = 10_000
    """The most candidates one request compares, so that one request cannot hold the server."""

    name: str = Field(min_length=1, max_length=NAME_MAX_LENGTH)
    """The name as written, e.g. ``ACME IND. (MTL)``; blank is no name."""
    candidates: list[Organization] = Field(max_length=CANDIDATES_MAX)
    """The known organizations it may be."""


class MatchVerdict(StrEnum):
    """The verdict on a name as a document wrote it."""

    MATCH = "match"
    """It is one of the candidates."""
    NEW_RECORD = "new_record"
    """It is none of them; a person creates it in the reference system."""
    UNCERTAIN = "uncertain"
    """Several candidates have the name; a person decides."""


class MatchDecision(BaseModel):
    """The verdict, and the candidates that have the written name."""

    model_config = ConfigDict(frozen=True)

    verdict: MatchVerdict
    """Whether one, none or several candidates have the name."""
    record_ids: list[str]
    """The ids of the candidates that have the name: one on a match, several when uncertain."""
