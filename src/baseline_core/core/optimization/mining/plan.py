"""A plan of a bound model, and the keys a planner's tables name its elements by."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from baseline_core.core.optimization.solver.search.kernel_state import KernelSnapshot, KernelState


if TYPE_CHECKING:
    from baseline_core.core.optimization.solver.kernel.kernel import Kernel


class Plan(BaseModel):
    """A start and a target per element, in the bound model's element order."""

    starts: list[int]
    """Per element, its start slot."""
    targets: list[int]
    """Per element, its target, as a position in the bound model's targets."""
    absent: list[int] = []
    """Elements left out: a search found them no place, or a table does not show them."""

    @classmethod
    def read(cls, kernel: "Kernel") -> "Plan":
        """Read the kernel's current placement, its absent elements named.

        Args:
            kernel: The bound kernel.

        Returns:
            The plan.
        """
        snapshot = KernelState(kernel).snapshot()
        absent = [element for element, out in enumerate(snapshot.absent) if out]
        return cls(starts=snapshot.starts, targets=snapshot.targets, absent=absent)

    def load(self, kernel: "Kernel") -> None:
        """Place every element of the kernel as the plan says, the absent ones left out.

        Args:
            kernel: The bound kernel.
        """
        absent = set(self.absent)
        mask = [int(element in absent) for element in range(len(self.starts))]
        KernelState(kernel).restore(KernelSnapshot(self.starts, self.targets, mask))


@dataclass(frozen=True, slots=True)
class PlanView:
    """How the planner's tables name one instance's elements and targets, and cut its days."""

    keys: dict[str, list[str]]
    """Per role fixed by the data (operator, product, lot…), each element's key."""
    target_role: str
    """The role read off the plan's targets (machine, vehicle…)."""
    target_names: list[str]
    """Per target position, its key in that role."""
    period: int
    """Slots in a day."""
    first_hour: int
    """The hour of a day's first slot."""
