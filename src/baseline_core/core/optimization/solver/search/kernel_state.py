"""The state a search runs on: the bound kernel, which holds the placement itself."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:
    from baseline_core.core.optimization.solver.kernel.kernel import Kernel


class KernelSnapshot(NamedTuple):
    """A copy of the kernel's placement and absent elements, to roll back to."""

    starts: list[int]
    """Per element, its start."""
    targets: list[int]
    """Per element, its target set."""
    absent: list[int]
    """Per element, a non-zero flag when it is left out."""


@dataclass(slots=True)
class ElementPlacement:
    """Where one element is: its start and target set, and whether it is left out."""

    element: int
    """The element."""
    start: int
    """Its start."""
    target: int
    """Its target set."""
    absent: int
    """A non-zero flag when it is left out."""


class KernelState:
    """The kernel a search edits, and a copy of its placement to roll back to."""

    def __init__(self, kernel: "Kernel") -> None:
        """Keep the kernel the moves edit.

        Args:
            kernel: The bound kernel.
        """
        self.kernel = kernel

    def snapshot(self) -> KernelSnapshot:
        """Copy the placement and the absent elements.

        Returns:
            The copy, which ``restore`` loads back.
        """
        starts, targets = self.kernel.placement
        return KernelSnapshot(starts, targets, self.kernel.absent_mask)

    def restore(self, snap: KernelSnapshot) -> None:
        """Load a copied placement and its absent elements back.

        Args:
            snap: A value ``snapshot`` returned, or one built from a plan.
        """
        self.kernel.set_absent(snap.absent)
        self.kernel.set_placement(snap.starts, snap.targets)

    def placements(self, elements: list[int]) -> list[ElementPlacement]:
        """Where each of the elements is.

        Args:
            elements: The elements.

        Returns:
            One placement per element, in their order.
        """
        starts, targets = self.kernel.placement
        absent = self.kernel.absent_mask
        return [ElementPlacement(j, starts[j], targets[j], absent[j]) for j in elements]

    def put(self, placements: list[ElementPlacement]) -> None:
        """Put each element where its placement says, left out when it was.

        Args:
            placements: The placements.
        """
        for each in placements:
            self.kernel.insert(each.element, each.start, each.target)

            if each.absent:
                self.kernel.remove(each.element)
