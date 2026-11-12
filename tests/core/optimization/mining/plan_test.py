from baseline_core.core.optimization.mining.plan import Plan
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel


class TestPlan:
    def test__a_loaded_plan__reads_back_whole_with_its_absent_elements(
        self, four_jobs: BoundModel
    ) -> None:
        """A plan loaded into the kernel reads back as it was, its absent elements named."""
        plan = Plan(starts=[0, 0, 2, 2], targets=[0, 1, 0, 1], absent=[1, 2])

        plan.load(four_jobs.kernel)

        assert Plan.read(four_jobs.kernel) == plan
