"""Release elements and re-place them one by one, each where it costs least."""

from dataclasses import dataclass
from typing import ClassVar

from numpy.random import Generator

from baseline_core.core.optimization.solver.search.kernel_state import (
    ElementPlacement,
    KernelState,
)
from baseline_core.core.optimization.solver.search.search_index import SearchIndex


@dataclass(frozen=True, slots=True)
class Release:
    """What a move releases, and how it re-places them."""

    elements: list[int]
    """The released elements, closed over what moves with them."""
    remove_first: bool
    """Whether every released element leaves the placement before any is re-placed."""
    in_order: bool = False
    """Whether they are re-placed in the order of their values, rather than tightest first."""


class DestroyRepair:
    """The released elements re-placed, a pair at one start, grouped pairs aligned."""

    PAIR_SIZE: ClassVar[int] = 2
    """The size of a group an alignment scans jointly."""

    def __init__(self, release: Release, index: SearchIndex, rng: Generator) -> None:
        """Keep the released elements and how they are re-placed.

        Args:
            release: The released elements, and how they are re-placed.
            index: The elements' placements and what moves together.
            rng: Draws among equally good placements.
        """
        self.indices = release.elements
        self.index = index
        self.rng = rng
        self.remove_first = release.remove_first
        self.in_order = release.in_order
        self.prior: list[ElementPlacement] = []
        """Where the released elements were before ``apply``."""

    def apply(self, state: KernelState) -> None:
        """Release the elements, re-place them, and align the grouped pairs still in conflict.

        Args:
            state: The kernel, edited in place.
        """
        kernel, index = state.kernel, self.index
        self.prior = state.placements(self.indices)

        if self.remove_first:
            for j in self.indices:
                kernel.remove(j)

        self._replace(state)

        if self.remove_first:
            # an element no placement took goes back where it was: the only spot known real
            left_out = kernel.absent_mask

            for each in self.prior:
                if left_out[each.element]:
                    kernel.insert(each.element, each.start, each.target)

        # grouped pairs need a joint scan only while a hard conflict is left
        if kernel.full_hard:
            released = set(self.indices)

            for group in index.groups:
                if len(group) == self.PAIR_SIZE and released.intersection(group):
                    self._align(state, group[0], group[1])

    def _replace(self, state: KernelState) -> None:
        """Re-place every released element, tightest first or in order, a pair in one scan.

        Args:
            state: The kernel, edited in place.
        """
        kernel, index = state.kernel, self.index
        released = set(self.indices)
        placed: set[int] = set()

        for j in self._in_value_order() if self.in_order else self._tightest_first():
            if j in placed:
                continue

            # a pair lands in one scan: alone, the first never sees where the second goes
            partner = index.partner[j]

            if partner >= 0 and partner in released and partner not in placed:
                kernel.relocate(j, partner, index.domains, self.rng.random())
                placed.add(partner)
            else:
                kernel.relocate(j, -1, index.domains, self.rng.random())

            placed.add(j)

    def revert(self, state: KernelState) -> None:
        """Put every released element back as it was.

        Args:
            state: The kernel, edited in place.
        """
        state.put(self.prior)
        self.prior = []

    def _in_value_order(self) -> list[int]:
        """The released elements in the order of the values they held, ties by element.

        Returns:
            The order they are re-placed in.
        """
        keyed = sorted([[each.start, each.element] for each in self.prior])
        return [element for _, element in keyed]

    def _tightest_first(self) -> list[int]:
        """The released elements, the tightest family first, then the longest element.

        Returns:
            The order they are re-placed in.
        """
        index = self.index
        # a family with few sets per element places before roomier ones crowd it out
        tightness = {
            family: sum(
                1.0 / max(1, len(index.sets(element))) for element in index.families[family]
            )
            if family >= 0
            else 0.0
            for family in {index.family_of[j] for j in self.indices}
        }

        keyed = sorted([[-tightness[index.family_of[j]], -index.span[j], j] for j in self.indices])
        return [int(j) for _, _, j in keyed]

    def _align(self, state: KernelState, first: int, second: int) -> None:
        """Scan both parts of a pair on one period until neither is in a hard conflict.

        Args:
            state: The kernel.
            first: The first part.
            second: The second part.
        """
        kernel, index = state.kernel, self.index

        if kernel.element_hard(first) + kernel.element_hard(second) == 0:
            return

        starts, targets = kernel.placement
        prior = [[first, starts[first], targets[first]], [second, starts[second], targets[second]]]

        by_period: dict[int, list[int]] = {}

        for start in index.starts(second):
            by_period.setdefault(start // index.period, []).append(start)

        for target_first in index.sets(first):
            for start_first in index.starts(first):
                kernel.insert(first, start_first, target_first)

                for start_second in by_period.get(start_first // index.period, []):
                    for target_second in index.sets(second):
                        kernel.insert(second, start_second, target_second)

                        if kernel.element_hard(first) + kernel.element_hard(second) == 0:
                            return

        for j, start, target in prior:
            kernel.insert(j, start, target)
