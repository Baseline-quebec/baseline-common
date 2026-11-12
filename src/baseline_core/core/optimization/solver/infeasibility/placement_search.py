"""Searching a placement of a group: exhaustively, to prove there is none, or heuristically."""

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from baseline_core.core.optimization.solver.kernel.kernel import (
        Kernel,
        SearchOutcome,
        SearchPlan,
    )
    from baseline_core.core.optimization.solver.stop_condition import StopCondition


class ExhaustiveSearch:
    """Every candidate, depth first, pruning forward: a dead end proves there is no placement."""

    def __init__(self, stop: "StopCondition") -> None:
        """Keep when the search gives up.

        Args:
            stop: Asked at its checkpoints; an iteration is one evaluation of the kernel.
        """
        self.stop = stop

    @staticmethod
    def reads(closed: list[int], ready: Callable[[int], bool]) -> list[int]:
        """Only the constraints its placed elements allow: one read too early would prove wrong.

        Args:
            closed: The constraints the group closes, as name positions.
            ready: Whether a constraint is read exactly once the placed elements are present.

        Returns:
            The ready ones.
        """
        return [index for index in closed if ready(index)]

    @staticmethod
    def ahead(closed: list[int], alone: Callable[[int], bool]) -> list[int]:
        """The constraints a later element allows alone: they prune its candidates.

        Args:
            closed: The constraints the group closes, as name positions.
            alone: Whether a constraint is read exactly around the later element alone.

        Returns:
            Those ones.
        """
        return [index for index in closed if alone(index)]

    def search(self, kernel: "Kernel", plan: "SearchPlan") -> "SearchOutcome":
        """The kernel's depth-first search, as it is: a dead end proves, a stop decides nothing.

        Args:
            kernel: The bound kernel, every element outside the group absent.
            plan: The group, its candidates and what each placement reads.

        Returns:
            What it found, the evaluations spent and what rejected candidates.
        """
        outcome: SearchOutcome = kernel.depth_first(plan, self.stop)
        return outcome


class HeuristicSearch:
    """A placement is looked for, depth first then by min-conflicts: a dead end proves nothing."""

    def __init__(self, stop: "StopCondition", seed: int) -> None:
        """Keep when the search gives up, and where its random draws start.

        Args:
            stop: Asked at its checkpoints; an iteration is one evaluation of the kernel.
            seed: The first draw of min-conflicts: the same seed, the same placement.
        """
        self.stop = stop
        self.seed = seed

    @staticmethod
    def reads(closed: list[int], ready: Callable[[int], bool]) -> list[int]:
        """Every constraint at once, never asked whether ready: the early reads only steer.

        Args:
            closed: The constraints the group closes, as name positions.
            ready: Not asked.

        Returns:
            All of them.
        """
        del ready
        return list(closed)

    @staticmethod
    def ahead(closed: list[int], alone: Callable[[int], bool]) -> list[int]:
        """Nothing is read forward: a heuristic search never prunes.

        Args:
            closed: Not read.
            alone: Not asked.

        Returns:
            None of them.
        """
        del closed, alone
        return []

    def search(self, kernel: "Kernel", plan: "SearchPlan") -> "SearchOutcome":
        """Depth first; when it stalls, min-conflicts tries too; only a placement decides.

        Args:
            kernel: The bound kernel, every element outside the group absent.
            plan: The group, its candidates and what each placement reads.

        Returns:
            Placeable when either found one, else undecided; the work and the blame of the first.
        """
        outcome: SearchOutcome = kernel.depth_first(plan, self.stop)
        found = outcome.found

        if found <= 0:
            # the depth-first pass can stall on an early choice: local search tries one too
            found = kernel.min_conflicts(plan, self.stop, self.seed)

        return replace(outcome, found=1 if found > 0 else -1)
