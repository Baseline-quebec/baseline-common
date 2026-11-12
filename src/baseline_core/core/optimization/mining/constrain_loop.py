"""Should the model be constrained more? Encode what the current model explains worst, repeat."""

from dataclasses import dataclass
from operator import itemgetter
from typing import Any

from pydantic import BaseModel, Field

from baseline_core.core.optimization.mining.candidate import Candidate, ConstrainedDefinition
from baseline_core.core.optimization.mining.plan import Plan, PlanView
from baseline_core.core.optimization.mining.sampler import AbstractPlanSampler
from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel, ModelCompiler


class ConstrainLoopConfig(BaseModel):
    """When the loop stops, and how each round's model is compiled."""

    level: float = Field(default=0.01, gt=0.0, lt=1.0)
    """Encode a candidate while the model matches the planner with less than this chance."""
    per_round: int = Field(default=1, gt=0)
    """Candidates encoded per round, the most broken first."""
    max_rounds: int = Field(default=50, gt=0)
    """Rounds at most."""
    cythonize: bool = True
    """Whether the kernels are compiled with Cython, as ``ModelCompiler`` takes it."""


@dataclass(frozen=True, slots=True)
class LoopRound:
    """One round: what the current model breaks, and what was encoded."""

    rates: dict[str, float]
    """Per remaining candidate, by name, the share of the model's plans that break it."""
    chances: dict[str, float]
    """Per remaining candidate, by name, the chance the model respects it as the planner does."""
    added: list[Candidate]
    """The candidates encoded at the end of the round."""


class ConstrainLoop:
    """Greedy: the candidate the known model explains worst is the one it lacks most."""

    def __init__(
        self,
        known: AbstractModelDefinition[Any],
        sampler: AbstractPlanSampler,
        config: ConstrainLoopConfig,
    ) -> None:
        """Keep the known model and how to run the rounds.

        Args:
            known: The model the accepted candidates are posted on.
            sampler: Draws the model's plans each round.
            config: When to stop, and how to compile.
        """
        self.known = known
        self.sampler = sampler
        self.config = config

    def run(
        self, problem: BaseModel, view: PlanView, candidates: list[Candidate]
    ) -> list[LoopRound]:
        """Run rounds until no candidate reaches the threshold, or ``max_rounds`` have run.

        Args:
            problem: The problem's data, as the known model reads it.
            view: How the planner's tables name the problem's elements.
            candidates: What the analysis of past plans found.

        Returns:
            Every round, in order.
        """
        current = self._bind(problem, view, [])
        self._check(current, view)
        judge = self._bind(problem, view, candidates)
        by_name = {candidate.name: candidate for candidate in candidates}
        accepted: list[Candidate] = []
        rounds: list[LoopRound] = []

        for _ in range(self.config.max_rounds):
            plans = self.sampler.sample(current)
            rest = [candidate for candidate in candidates if candidate not in accepted]
            rates = self._broken_rates(judge, plans, rest)
            chances = dict(
                sorted(((c.name, c.chance(rates[c.name])) for c in rest), key=itemgetter(1))
            )
            added = [
                by_name[name]
                for name, chance in list(chances.items())[: self.config.per_round]
                if chance < self.config.level
            ]
            rounds.append(LoopRound(rates=rates, chances=chances, added=added))

            if not added:
                break

            accepted += added
            current = self._bind(problem, view, accepted)

        return rounds

    def _bind(self, problem: BaseModel, view: PlanView, candidates: list[Candidate]) -> BoundModel:
        """Bind the known model with candidates posted on it.

        Args:
            problem: The problem's data.
            view: How the candidates name the elements.
            candidates: The candidates to post.

        Returns:
            The bound model.
        """
        definition = ConstrainedDefinition(self.known, view, candidates)
        compiler = ModelCompiler(definition, cythonize=self.config.cythonize)
        return compiler.bind(problem)

    @staticmethod
    def _check(bound: BoundModel, view: PlanView) -> None:
        """Refuse a view that names another instance's elements or targets.

        Args:
            bound: The model bound to the problem.
            view: The view handed with it.

        Raises:
            ValueError: A role's keys or the target names do not match the bound model.
        """
        elements, targets = bound.kernel.n, len(bound.data.targets)
        sizes = {role: len(keys) for role, keys in view.keys.items()}

        if any(size != elements for size in sizes.values()) or (
            targets and len(view.target_names) != targets
        ):
            msg = (
                f"the view names {sizes} elements and {len(view.target_names)} targets;"
                f" the problem has {elements} and {targets}"
            )
            raise ValueError(msg)

    @staticmethod
    def _broken_rates(
        judge: BoundModel, plans: list[Plan], candidates: list[Candidate]
    ) -> dict[str, float]:
        """Count, per candidate, the share of plans that break it, read in the judge's kernel.

        Args:
            judge: The known model with every candidate posted, each a rule named after it.
            plans: The plans to judge.
            candidates: The candidates to read.

        Returns:
            Per candidate, by name, the share of plans that break it.
        """
        kernel = judge.kernel
        index = {name: position for position, name in enumerate(kernel.names)}
        broken = dict.fromkeys((candidate.name for candidate in candidates), 0)

        for plan in plans:
            plan.load(kernel)

            for name in broken:
                broken[name] += int(kernel.full_count(index[name]) > 0)

        return {name: count / len(plans) for name, count in broken.items()}
