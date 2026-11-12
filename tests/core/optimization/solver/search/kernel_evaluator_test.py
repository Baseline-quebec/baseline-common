from typing import ClassVar

from numpy.random import default_rng

from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel, ModelCompiler
from baseline_core.core.optimization.solver.score import Score
from baseline_core.core.optimization.solver.search.destroy_repair import DestroyRepair, Release
from baseline_core.core.optimization.solver.search.kernel_evaluator import KernelEvaluator
from baseline_core.core.optimization.solver.search.kernel_state import KernelState
from baseline_core.core.optimization.solver.search.search_index import SearchIndex
from tests.core.optimization.bin_packing_model import Fleet


class TestApply:
    MOVES: ClassVar[int] = 200
    """How many random moves each model is checked on."""

    @classmethod
    def mismatches(cls, bound: BoundModel) -> int:
        """How many random moves report a change that a full recount does not see.

        Args:
            bound: The bound model, placed.

        Returns:
            The count.
        """
        kernel = bound.kernel
        kernel.soft_mode = True
        evaluator, state, rng = KernelEvaluator(bound), KernelState(kernel), default_rng(0)
        index = SearchIndex.of(bound)
        found = 0

        for _ in range(cls.MOVES):
            size = int(rng.integers(1, min(kernel.n, 4) + 1))
            picked = sorted(int(j) for j in rng.choice(kernel.n, size=size, replace=False))
            release = Release(
                picked, remove_first=bool(rng.integers(2)), in_order=bool(rng.integers(2))
            )
            move = DestroyRepair(release, index, rng)
            hard, soft = kernel.full_hard, kernel.full_soft

            change = evaluator.apply(move, state)

            found += change != Score(hard=kernel.full_hard - hard, soft=kernel.full_soft - soft)

        return found

    def test__a_move_between_sets__changes_the_score_by_what_a_full_recount_sees(
        self, overloaded: BoundModel
    ) -> None:
        """A parcel leaving its truck changes the rule of every parcel left behind, counted."""
        assert self.mismatches(overloaded) == 0

    def test__a_compiled_kernel__counts_a_move_between_sets_as_a_full_recount_does(
        self, compiled_stacking: ModelCompiler[Fleet], fleet: Fleet
    ) -> None:
        """The set an element left is read by the compiled kernel too."""
        bound = compiled_stacking.bind(fleet)
        bound.kernel.set_placement([0] * 9, [0] * 9)

        assert self.mismatches(bound) == 0

    def test__a_move_in_the_job_shop__changes_the_score_by_what_a_full_recount_sees(
        self, shop: BoundModel
    ) -> None:
        """Operations moved between machines and starts are scored as a full recount does."""
        assert self.mismatches(shop) == 0

    def test__a_move_of_the_crew__changes_the_score_by_what_a_full_recount_sees(
        self, bound: BoundModel
    ) -> None:
        """Tasks moved between stations and starts are scored as a full recount does."""
        assert self.mismatches(bound) == 0

    def test__a_move__leaves_the_placement_the_move_alone_leaves(
        self, overloaded: BoundModel, stacking: ModelCompiler[Fleet], fleet: Fleet
    ) -> None:
        """The old placement comes back only for the count: the move's own result stays."""
        alone = stacking.bind(fleet)
        alone.kernel.set_placement([0] * 9, [0] * 9)
        moves = [
            DestroyRepair(
                Release([0, 1, 2], remove_first=True), SearchIndex.of(each), default_rng(1)
            )
            for each in [overloaded, alone]
        ]

        KernelEvaluator(overloaded).apply(moves[0], KernelState(overloaded.kernel))
        moves[1].apply(KernelState(alone.kernel))

        assert overloaded.kernel.placement == alone.kernel.placement
        assert overloaded.kernel.absent_mask == alone.kernel.absent_mask
        assert overloaded.kernel.placement != [[0] * 9, [0] * 9]
