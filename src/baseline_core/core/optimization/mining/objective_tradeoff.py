"""Two objectives: which way did the plans move from plans that only respect the rules?"""

import math
from statistics import fmean, pstdev

from baseline_core.core.optimization.mining.plan import Plan
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.search.kernel_state import KernelState


class ObjectiveTradeoff:
    """The direction of improvement on two objective terms of the model, as an angle."""

    def __init__(self, terms: list[str]) -> None:
        """Keep the two objective terms, by their names in the model.

        Args:
            terms: ``[first, second]``.

        Raises:
            ValueError: ``terms`` does not name two distinct terms.
        """
        if len(set(terms)) != 2 or len(terms) != 2:  # noqa: PLR2004  a trade-off is between two
            msg = f"a trade-off is between two distinct terms; got {terms}"
            raise ValueError(msg)

        self.terms = terms

    def measure(self, bound: BoundModel, plans: list[Plan]) -> list[list[int]]:
        """Count each term on each plan, the kernel left as it was given.

        Args:
            bound: The model whose kernel loads each plan.
            plans: The plans to measure.

        Returns:
            One row per plan, one raw count per term.
        """
        kernel = bound.kernel
        state = KernelState(kernel)
        snapshot = state.snapshot()
        positions = [kernel.names.index(name) for name in self.terms]
        rows = []

        try:
            for plan in plans:
                plan.load(kernel)
                rows.append([int(kernel.full_count(position)) for position in positions])
        finally:
            state.restore(snapshot)

        return rows

    @staticmethod
    def angle(baseline: list[list[int]], measured: list[list[int]]) -> float:
        """Degrees between the first term's axis and the mean improvement over ``baseline``.

        Args:
            baseline: The terms measured on plans that respect the rules only.
            measured: The terms measured on the plans to place.

        Returns:
            The angle, each axis in units of the baseline's spread.
        """
        gain = []

        for term in range(2):
            before = [row[term] for row in baseline]
            after = [row[term] for row in measured]
            gain.append((fmean(before) - fmean(after)) / (pstdev(before) + 1e-9))

        return math.degrees(math.atan2(gain[1], gain[0]))
