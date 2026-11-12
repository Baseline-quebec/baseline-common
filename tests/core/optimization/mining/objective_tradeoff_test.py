import math

import pytest

from baseline_core.core.optimization.mining.objective_tradeoff import ObjectiveTradeoff
from baseline_core.core.optimization.mining.plan import Plan
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel


class TestObjectiveTradeoff:
    def test__measure__reads_each_term_on_each_plan_and_restores_the_kernel(
        self, four_jobs: BoundModel
    ) -> None:
        """Lateness sums the starts, machine use counts the jobs on B; the kernel ends as given."""
        before = four_jobs.kernel.placement
        plans = [
            Plan(starts=[0, 0, 2, 2], targets=[0, 1, 0, 1]),
            Plan(starts=[0, 2, 4, 6], targets=[0, 0, 0, 0]),
        ]

        rows = ObjectiveTradeoff(["lateness", "machine_use"]).measure(four_jobs, plans)

        assert rows == [[4, 2], [12, 0]]
        assert four_jobs.kernel.placement == before

    @pytest.mark.parametrize(
        ("measured", "degrees"),
        [([[8, 11], [10, 11]], 0.0), ([[11, 8], [11, 10]], 90.0), ([[8, 8], [10, 10]], 45.0)],
    )
    def test__the_angle__points_at_the_term_that_improved(
        self, measured: list[list[int]], degrees: float
    ) -> None:
        """Only the first term lower: 0°; only the second: 90°; both alike: 45°."""
        baseline = [[10, 10], [12, 12]]

        angle = ObjectiveTradeoff(["first", "second"]).angle(baseline, measured)

        assert math.isclose(angle, degrees, abs_tol=1e-6)

    def test__one_term__is_refused(self) -> None:
        """A trade-off needs two distinct terms."""
        with pytest.raises(ValueError, match="two distinct terms"):
            ObjectiveTradeoff(["lateness", "lateness"])
