"""A move scored around the elements it moved, against the running score of the placement."""

import numpy as np

from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.score import Score
from baseline_core.core.optimization.solver.search.destroy_repair import DestroyRepair
from baseline_core.core.optimization.solver.search.kernel_state import KernelState


class KernelEvaluator:
    """Every constraint counted around the moved elements before and after, weighted per axis."""

    def __init__(self, bound: BoundModel) -> None:
        """Keep the kernel and each constraint's axis and weight, then score the placement.

        Args:
            bound: The bound model the search runs on.
        """
        self.kernel = bound.kernel
        self.hard = bound.model.hard
        self.weights = bound.model.unit_weights
        self.current = Score(hard=self.kernel.full_hard, soft=self.kernel.full_soft)
        """The score of the placement, advanced only by ``commit``."""

    def apply(self, move: DestroyRepair, state: KernelState) -> Score:
        """Apply a move and return its change, counted on the same instances before and after.

        Args:
            move: The move; its released elements hold everything it moves.
            state: The placement, edited in place.

        Returns:
            The change.
        """
        move.apply(state)
        elements, prior = move.indices, move.prior
        moved = state.placements(elements)
        changed = [k for k in range(len(elements)) if prior[k] != moved[k]]

        after = self.counts(elements, [each.target for each in prior])
        state.put([prior[k] for k in changed])
        before = self.counts(elements, [each.target for each in moved])
        state.put([moved[k] for k in changed])

        return self.change(before, after)

    def counts(self, elements: list[int], others: list[int]) -> list[int]:
        """Every constraint's raw count around the elements.

        Args:
            elements: The elements.
            others: Per element, another set whose instances it can change, or -1.

        Returns:
            One count per constraint.
        """
        around = np.asarray(elements, dtype=np.int64)
        sets = np.asarray(others, dtype=np.int64)
        return [
            self.kernel.around_set_count(index, around, sets) for index in range(len(self.hard))
        ]

    def change(self, before: list[int], after: list[int]) -> Score:
        """The weighted change between two counts of every constraint, hard and soft apart.

        Args:
            before: The counts before the move.
            after: The counts after it.

        Returns:
            The change.
        """
        hard = soft = 0

        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if self.hard[index]:
                hard += self.weights[index] * (new - old)
            else:
                soft += self.weights[index] * (new - old)

        return Score(hard=hard, soft=soft)

    def commit(self, change: Score) -> None:
        """Advance the running score by an accepted move's change.

        Args:
            change: The move's change.
        """
        self.current += change
