"""Ways to draw reference plans from a bound model: what the known model produces on its own."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from numpy.random import Generator, default_rng
from pydantic import BaseModel, Field

from baseline_core.core.optimization.mining.plan import Plan
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.search.kernel_solver import (
    KernelSolver,
    KernelSolverConfig,
)
from baseline_core.core.optimization.solver.search.kernel_state import KernelState
from baseline_core.core.optimization.solver.search.search_index import SearchIndex


if TYPE_CHECKING:
    from baseline_core.core.optimization.solver.kernel.kernel import Kernel


class AbstractPlanSampler(ABC):
    """Draws plans from a bound model; its kernel ends as it was given."""

    def sample(self, bound: BoundModel) -> list[Plan]:
        """Draw the plans, then restore the kernel's placement, absences and score mode.

        Args:
            bound: The model bound to its data.

        Returns:
            The plans drawn.
        """
        state, soft_mode = KernelState(bound.kernel), bound.kernel.soft_mode
        snapshot = state.snapshot()

        try:
            return self._sample(bound)
        finally:
            state.restore(snapshot)
            bound.kernel.soft_mode = soft_mode

    @abstractmethod
    def _sample(self, bound: BoundModel) -> list[Plan]:
        """Draw the plans, free to move the kernel.

        Args:
            bound: The model bound to its data.

        Returns:
            The plans drawn.
        """


class FeasibleWalkConfig(BaseModel):
    """A random walk over plans the hard rules allow, with no objective."""

    samples: int = Field(default=200, gt=0)
    """Plans kept."""
    burn_in: int = Field(default=40_000, ge=0)
    """Moves before the first plan is kept."""
    thinning: int = Field(default=3_000, gt=0)
    """Moves between two kept plans."""
    max_thinnings: int = Field(default=2_000, gt=0)
    """Thinning rounds at most, feasible or not, before the walk gives up."""
    seed: int = 7
    """The seed of the random moves."""


@dataclass(frozen=True, slots=True)
class RandomWalk:
    """One element relocated at a time inside its domain, a move kept when no worse."""

    kernel: "Kernel"
    """The bound kernel the walk moves."""
    index: SearchIndex
    """Each element's domain."""
    movable: list[int]
    """The present elements with a non-empty domain."""
    rng: Generator
    """The walk's random draws."""

    def run(self, steps: int) -> None:
        """Try random relocations, keeping each that does not worsen the score.

        Args:
            steps: How many relocations to try.
        """
        for _ in range(steps):
            element = self.movable[self.rng.integers(len(self.movable))]
            starts, sets = self.index.starts(element), self.index.sets(element)
            start = starts[self.rng.integers(len(starts))]
            target = sets[self.rng.integers(len(sets))]

            if self.kernel.delta_place(element, start, target) <= [0, 0]:
                self.kernel.insert(element, start, target)


class FeasibleWalk(AbstractPlanSampler):
    """Plans the known hard rules allow, drawn with no preference, from the kernel's placement."""

    def __init__(self, config: FeasibleWalkConfig) -> None:
        """Keep the settings.

        Args:
            config: The walk's length, thinning, cap and seed.
        """
        self.config = config

    @override
    def _sample(self, bound: BoundModel) -> list[Plan]:
        """Walk on the hard score only, keeping a feasible plan every ``thinning`` moves.

        Args:
            bound: The model bound to its data, its kernel where the walk starts.

        Returns:
            ``samples`` feasible plans.

        Raises:
            RuntimeError: The walk did not find them within ``max_thinnings`` rounds.
        """
        kernel, config = bound.kernel, self.config
        index = SearchIndex.of(bound)
        absent = kernel.absent_mask
        movable = [
            element
            for element in range(kernel.n)
            if not absent[element] and index.starts(element) and index.sets(element)
        ]
        walk = RandomWalk(kernel, index, movable, default_rng(config.seed))
        kernel.soft_mode = False
        walk.run(config.burn_in)
        plans: list[Plan] = []

        for _ in range(config.max_thinnings):
            walk.run(config.thinning)

            if kernel.full_hard == 0:
                plans.append(Plan.read(kernel))

            if len(plans) == config.samples:
                return plans

        kept, rounds = f"{len(plans)} of {config.samples}", config.max_thinnings
        msg = f"the walk kept {kept} plans in {rounds} rounds"
        raise RuntimeError(msg)


class KernelRestartsConfig(BaseModel):
    """Plans of the model's own destroy-and-repair search, one per seed."""

    samples: int = Field(default=60, gt=0)
    """Restarts."""
    solver: KernelSolverConfig = KernelSolverConfig(seed=1000)
    """The first restart's settings; each next restart adds one to the seed."""


class KernelRestarts(AbstractPlanSampler):
    """Plans the known model's search produces on its own; what it leaves out stays absent."""

    def __init__(self, config: KernelRestartsConfig) -> None:
        """Keep the settings.

        Args:
            config: The restarts and the solver's settings.
        """
        self.config = config

    @override
    def _sample(self, bound: BoundModel) -> list[Plan]:
        """One search per seed, read off the kernel with the elements it left out.

        Args:
            bound: The model bound to its data.

        Returns:
            ``samples`` plans.
        """
        plans: list[Plan] = []

        for restart in range(self.config.samples):
            seed = self.config.solver.seed + restart
            KernelSolver(self.config.solver.model_copy(update={"seed": seed})).solve(bound)
            plans.append(Plan.read(bound.kernel))

        return plans
