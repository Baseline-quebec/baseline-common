import pytest

from baseline_core.core.optimization.mining.candidate import Candidate, CandidateFeature
from baseline_core.core.optimization.mining.constrain_loop import (
    ConstrainLoop,
    ConstrainLoopConfig,
)
from baseline_core.core.optimization.mining.sampler import KernelRestarts, KernelRestartsConfig
from baseline_core.core.optimization.solver.search.kernel_solver import KernelSolverConfig
from tests.core.optimization.mining.shop import Shop, ShopDefinition


class TestConstrainLoop:
    @pytest.fixture
    def loop(self) -> ConstrainLoop:
        """The loop on the shop, three short solves a round, its kernels run as plain Python.

        Returns:
            The loop.
        """
        solver = KernelSolverConfig(n_iters=200, patience=50, seed=1)
        return ConstrainLoop(
            ShopDefinition(),
            KernelRestarts(KernelRestartsConfig(samples=3, solver=solver)),
            ConstrainLoopConfig(cythonize=False),
        )

    @pytest.fixture
    def two_jobs(self) -> Shop:
        """A shop of one ``a`` job and one ``b`` job.

        Returns:
            The shop.
        """
        return Shop.of(["a", "b"])

    def test__the_loop__encodes_what_the_model_breaks_then_stops(
        self, loop: ConstrainLoop, two_jobs: Shop
    ) -> None:
        """The model puts ``a`` on day 0: « a not on day 0 » is encoded, « b not on day 1 » not."""
        a_off, b_off = [
            Candidate(
                role="op",
                key=key,
                feature=CandidateFeature.UNUSED_DAY,
                value=day,
                kept=12,
                plans=12,
            )
            for key, day in {"a": 0, "b": 1}.items()
        ]

        rounds = loop.run(two_jobs, two_jobs.view, [a_off, b_off])

        assert [[candidate.key for candidate in r.added] for r in rounds] == [["a"], []]

    def test__a_view_of_another_instance__is_refused(
        self, loop: ConstrainLoop, two_jobs: Shop
    ) -> None:
        """A view naming three jobs cannot read a shop of two."""
        view = Shop.of(["a", "b", "c"]).view

        with pytest.raises(ValueError, match="the view names"):
            loop.run(two_jobs, view, [])
