import pytest

from baseline_core.core.optimization.mining.plan import Plan
from baseline_core.core.optimization.mining.sampler import (
    FeasibleWalk,
    FeasibleWalkConfig,
    KernelRestarts,
    KernelRestartsConfig,
)
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel, ModelCompiler
from baseline_core.core.optimization.solver.search.kernel_solver import KernelSolverConfig
from tests.core.optimization.mining.shop import Shop


class TestKernelRestarts:
    def test__the_solver_s_plans__are_its_optima_and_the_kernel_is_restored(
        self, four_jobs: BoundModel
    ) -> None:
        """Each restart's plan uses both machines from slot 0; the kernel ends as it was given."""
        kernel = four_jobs.kernel
        before = [kernel.placement, kernel.absent_mask, kernel.soft_mode]
        config = KernelRestartsConfig(samples=2, solver=KernelSolverConfig(seed=3))

        plans = KernelRestarts(config).sample(four_jobs)

        assert [sorted(plan.starts) for plan in plans] == [[0, 0, 2, 2]] * 2
        assert [kernel.placement, kernel.absent_mask, kernel.soft_mode] == before


class TestFeasibleWalk:
    @staticmethod
    def overlaps(plan: Plan) -> bool:
        """Tell whether two present jobs of the plan share a machine slot.

        Args:
            plan: The plan to check.

        Returns:
            ``True`` when two present jobs share a machine slot.
        """
        present = [j for j in range(len(plan.starts)) if j not in plan.absent]
        cells = [(plan.targets[j], plan.starts[j] + k) for j in present for k in range(2)]
        return len(cells) != len(set(cells))

    def test__a_walk__keeps_only_feasible_plans_and_restores_the_kernel(
        self, four_jobs: BoundModel
    ) -> None:
        """From the bind's placement, every kept plan has no overlap; the kernel ends as given."""
        kernel = four_jobs.kernel
        before = [kernel.placement, kernel.absent_mask, kernel.soft_mode]
        config = FeasibleWalkConfig(samples=5, burn_in=500, thinning=50, seed=1)

        plans = FeasibleWalk(config).sample(four_jobs)

        assert len(plans) == 5
        assert not any(self.overlaps(plan) for plan in plans)
        assert [kernel.placement, kernel.absent_mask, kernel.soft_mode] == before

    def test__a_walk_from_a_loaded_plan__keeps_its_absent_element_out(
        self, four_jobs: BoundModel
    ) -> None:
        """The walk starts where the kernel is: the absent job stays out and in place."""
        Plan(starts=[0, 0, 2, 7], targets=[0, 1, 0, 1], absent=[3]).load(four_jobs.kernel)
        config = FeasibleWalkConfig(samples=4, burn_in=200, thinning=50, seed=2)

        plans = FeasibleWalk(config).sample(four_jobs)

        assert all(plan.absent == [3] and plan.starts[3] == 7 for plan in plans)
        assert not any(self.overlaps(plan) for plan in plans)

    def test__a_walk_that_finds_no_feasible_plan__gives_up(
        self, shop_compiler: ModelCompiler[Shop]
    ) -> None:
        """Seventeen two-slot jobs cannot fit two machines of sixteen slots: the walk raises."""
        crowded = shop_compiler.bind(Shop.of(["a"] * 17))
        config = FeasibleWalkConfig(samples=1, burn_in=0, thinning=10, max_thinnings=3)

        with pytest.raises(RuntimeError, match="kept 0 of 1 plans in 3 rounds"):
            FeasibleWalk(config).sample(crowded)
