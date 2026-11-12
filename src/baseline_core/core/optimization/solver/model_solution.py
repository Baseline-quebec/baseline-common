"""What a solve takes and returns, in keys only: assignments, elements left out, the summary."""

from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from baseline_core.core.optimization.solver.score import Score


class VariableAssignment(BaseModel):
    """Values given to one element's decision variables, by key; a ``None`` one is left free."""

    model_config = ConfigDict(frozen=True)

    element: str
    """The element, by the key of the decided set."""
    value: int | None = None
    """Its integer variable; ``None`` leaves it free, or the model declares none."""
    target: str | None = None
    """Its target, by the key of the target set; ``None`` leaves it free or assigns nothing."""

    @model_validator(mode="after")
    def _gives_a_value(self) -> Self:
        """Refuse an assignment that gives neither a value nor a target.

        Returns:
            The assignment.

        Raises:
            ValueError: Both are ``None``.
        """
        if self.value is None and self.target is None:
            msg = f"the assignment of {self.element!r} gives neither a value nor a target"
            raise ValueError(msg)

        return self


@dataclass(frozen=True, slots=True)
class TermSummary:
    """What one minimized term costs in a solution."""

    value: int
    """Its count."""
    weight: int
    """Its weight in the model."""
    cost: int
    """Its count times its weight."""


@dataclass(frozen=True, slots=True)
class Summary:
    """What the model counts on a solution: the same shape for every model, nothing mapped."""

    score: Score
    """The weighted hard and soft scores, compared hard first; feasible when hard is 0."""
    terms: dict[str, TermSummary]
    """One per ``model.minimize(name=...)``."""
    constraints: dict[str, int]
    """One per ``model.constraint(name=...)``: its violations."""


class ModelSolution(BaseModel):
    """A solution of a bound model: its assignments, the elements left out, the summary."""

    model_config = ConfigDict(frozen=True)

    assignments: list[VariableAssignment]
    """One per present element, in position order: what a run stores and a solve takes back."""
    unassigned: list[str]
    """The keys of the absent elements."""
    summary: Summary
    """What the model counts on the assignments."""
