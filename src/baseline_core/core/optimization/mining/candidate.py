"""A rule mined from the planner's plans, and the hard rule it posts on a model."""

import math
from enum import StrEnum
from typing import TYPE_CHECKING, Any, override

from pydantic import BaseModel

from baseline_core.core.optimization.mining.plan import PlanView
from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.declaration import IntegerDecision, Member
from baseline_core.core.optimization.problem_formulation.expression import Expr
from baseline_core.core.optimization.problem_formulation.model import Model


if TYPE_CHECKING:
    from baseline_core.core.optimization.problem_formulation.declaration import Collection


class CandidateFeature(StrEnum):
    """What a candidate bounds, as the analysis of past plans names it.

    Attributes:
        EARLIEST_START: The key never starts before ``value`` (an hour).
        LATEST_END: The key never ends after ``value`` (an hour).
        MAX_PER_DAY: The key never has more than ``value`` elements in a day.
        UNUSED_DAY: The key is never placed on day ``value``.
        CLOSED_HOURS: Nothing is placed on day ``value`` during ``hours``.
    """

    EARLIEST_START = "earliest_start"
    LATEST_END = "latest_end"
    MAX_PER_DAY = "max_per_day"
    UNUSED_DAY = "unused_day"
    CLOSED_HOURS = "closed_hours"


class Candidate(BaseModel):
    """One bound on one key of one role, as the analysis reported it."""

    role: str
    """The role, by its column name in the planner's tables; ``*`` for everything."""
    key: str
    """The key in that role; ``*`` for everything."""
    feature: CandidateFeature
    """What the candidate bounds."""
    value: int
    """The hour, the count or the day index, as ``feature`` says."""
    hours: list[int] = []
    """For ``CLOSED_HOURS``, the hours closed."""
    kept: int
    """How many of the planner's plans respect it."""
    plans: int
    """How many plans the planner handed over."""

    @property
    def name(self) -> str:
        """The candidate in one line: the name of the rule it posts."""
        extra = f" {self.hours}" if self.hours else ""
        return f"{self.role} {self.key} {self.feature} {self.value}{extra}"

    def chance(self, broken_share: float) -> float:
        """The chance that plans breaking it this often respect it as often as the planner's.

        Args:
            broken_share: The share of the current model's plans that break it.

        Returns:
            The chance of at least ``kept`` respecting plans out of ``plans``.
        """
        keep = 1 - broken_share
        return float(
            sum(
                math.comb(self.plans, respected)
                * keep**respected
                * broken_share ** (self.plans - respected)
                for respected in range(self.kept, self.plans + 1)
            )
        )

    def post(self, model: Model[Any], view: PlanView) -> None:
        """Post the candidate as a hard rule, named after it, on the elements its key names.

        Args:
            model: The model, its decisions already declared.
            view: Each element's keys, and the day's slots.

        Raises:
            ValueError: The model has no integer decision, or no assignment for a target's key.
        """
        start, elements = model.integer_decision, model.decided_set

        if start is None or elements is None:
            msg = "a candidate bounds an integer decision: the model declares none"
            raise ValueError(msg)

        self._check(view)

        if self.role != view.target_role:
            named: Collection[Any] = elements

            if self.role != "*":
                flags = [int(key == self.key) for key in view.keys[self.role]]
                named = elements.where(model.lookup_table(flags))

            rule = self._rule(start, view, [named.member, named.member])
            model.constraint(rule, name=self.name)
            return

        if model.assignment is None:
            msg = f"{self.name} bounds a target: the model assigns none"
            raise ValueError(msg)

        flags = [int(name == self.key) for name in view.target_names]
        named_target = model.lookup_table(flags)

        for held in model.assignment.groups:
            rule = self._rule(start, view, [held.member, held.member])
            model.constraint((named_target[held.target] == 0) | rule, name=self.name)

    def _check(self, view: PlanView) -> None:
        """Refuse a candidate whose role or key the view does not hold.

        Args:
            view: How the tables name the instance's elements and targets.

        Raises:
            ValueError: The role is unknown, or no element or target carries the key.
        """
        if self.role == "*":
            return

        if self.role == view.target_role:
            keys = view.target_names
        elif self.role in view.keys:
            keys = view.keys[self.role]
        else:
            roles = [*sorted(view.keys), view.target_role]
            msg = f"{self.name}: the view has no role {self.role!r}; it has {roles}"
            raise ValueError(msg)

        if self.key not in keys:
            msg = f"{self.name}: nothing in the instance carries the key {self.key!r}"
            raise ValueError(msg)

    def _rule(self, start: IntegerDecision, view: PlanView, members: list[Member[Any]]) -> Expr:
        """Write the bound on one member, counted among the others where the feature counts.

        Args:
            start: The model's integer decision.
            view: The day's slots.
            members: ``[the member bounded, another member of the same collection]``.

        Returns:
            The rule's condition.
        """
        block, other = members
        period, first_hour = view.period, view.first_hour
        day = start[block] // period

        if self.feature == CandidateFeature.EARLIEST_START:
            return start[block] - day * period >= self.value - first_hour

        if self.feature == CandidateFeature.LATEST_END:
            return start.end(block) - day * period <= self.value - first_hour

        if self.feature == CandidateFeature.UNUSED_DAY:
            return day != self.value

        if self.feature == CandidateFeature.MAX_PER_DAY:
            return Model.count(other, start[other] // period == day) <= self.value

        runs: list[list[int]] = []

        for hour in sorted(self.hours):
            if runs and runs[-1][-1] == hour - 1:
                runs[-1].append(hour)
            else:
                runs.append([hour])

        offset = self.value * period - first_hour
        return Model.forall([
            (start.end(block) <= offset + run[0]) | (start[block] > offset + run[-1])
            for run in runs
        ])


class ConstrainedDefinition[S: BaseModel](AbstractModelDefinition[S]):
    """A known model, with candidates posted on top of it as hard rules."""

    def __init__(
        self, known: AbstractModelDefinition[S], view: PlanView, candidates: list[Candidate]
    ) -> None:
        """Keep the known model and what to post on it.

        Args:
            known: The model the candidates are added to.
            view: Each element's keys, as the candidates name them.
            candidates: The candidates to post.
        """
        self.schema = known.schema
        self.known = known
        self.view = view
        self.candidates = candidates

    @override
    def define(self, model: Model[S]) -> None:
        """Post the known model's rules, then one rule per candidate.

        Args:
            model: The model to post on.
        """
        self.known.define(model)

        for candidate in self.candidates:
            candidate.post(model, self.view)
