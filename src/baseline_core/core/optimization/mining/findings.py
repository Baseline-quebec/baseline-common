"""The ``findings.json`` an analysis of past plans writes, read into candidates."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from baseline_core.core.optimization.mining.candidate import Candidate, CandidateFeature


class Respected(BaseModel):
    """How many of the past plans respect a finding, written ``kept/plans``."""

    kept: int
    """Plans that respect it."""
    plans: int
    """Plans handed over."""

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, value: Any) -> Any:  # noqa: ANN401  pydantic hands over the raw input
        """Read the ``kept/plans`` text the analysis writes.

        Args:
            value: The raw value.

        Returns:
            The fields, or ``value`` untouched when it is not text.
        """
        if not isinstance(value, str):
            return value

        kept, plans = value.split("/")
        return {"kept": kept, "plans": plans}


class ProfileRow(BaseModel):
    """A bound on one key, one row of the per-key bounds table (``T17_profils``)."""

    model_config = ConfigDict(populate_by_name=True)

    role: str
    """The role, by its column name in the plans' tables."""
    key: str = Field(alias="cle")
    """The key in that role."""
    feature: CandidateFeature
    """What the row bounds."""
    bound: int
    """The hour, the count or the day index."""
    respected: Respected = Field(alias="plans_qui_respectent")
    """How many plans respect it."""

    @property
    def candidate(self) -> Candidate:
        """The row, as a candidate."""
        return Candidate(
            role=self.role,
            key=self.key,
            feature=self.feature,
            value=self.bound,
            kept=self.respected.kept,
            plans=self.respected.plans,
        )


class Profiles(BaseModel):
    """The per-key bounds table: the rows the analysis takes for rules, and those it asks about."""

    rules: list[ProfileRow] = Field(alias="regles_probables")
    """Rows it takes for rules."""
    questions: list[ProfileRow] = Field(alias="regle_ou_habitude")
    """Rows that may be a rule or a habit."""


class EmptyCells(BaseModel):
    """The day-and-hour cells no plan uses (``T1_recurrent_empty_cells``)."""

    cells: list[list[str]] = Field(alias="recurrent_empty_cells")
    """``[day label, hour]`` per cell."""
    respected: Respected = Field(alias="plans_exiges")
    """How many plans leave them empty."""

    @field_validator("cells", mode="before")
    @classmethod
    def _as_text(cls, cells: list[list[Any]]) -> list[list[str]]:
        """Read every cell's day and hour as text.

        Args:
            cells: The raw cells.

        Returns:
            The cells.
        """
        return [[str(day), str(hour)] for day, hour in cells]


class Findings(BaseModel):
    """An analysis's findings, as much of them as the mining reads."""

    profiles: Profiles = Field(alias="T17_profils")
    """The per-key bounds."""
    empty: EmptyCells = Field(alias="T1_recurrent_empty_cells")
    """The cells no plan uses."""

    def rules(self, days: list[str]) -> list[Candidate]:
        """Read the findings the analysis takes for rules: per-key bounds, then closed hours.

        Args:
            days: The day labels, in order.

        Returns:
            One candidate per bound taken for a rule, then one per day with recurrent empty hours.
        """
        closed: dict[int, list[int]] = {}

        for day, hour in self.empty.cells:
            closed.setdefault(days.index(day), []).append(int(hour))

        return [row.candidate for row in self.profiles.rules] + [
            Candidate(
                role="*",
                key="*",
                feature=CandidateFeature.CLOSED_HOURS,
                value=day,
                hours=sorted(hours),
                kept=self.empty.respected.kept,
                plans=self.empty.respected.plans,
            )
            for day, hours in sorted(closed.items())
        ]

    @property
    def questions(self) -> list[Candidate]:
        """The findings the analysis asks about: one candidate per bound it doubts."""
        return [row.candidate for row in self.profiles.questions]
