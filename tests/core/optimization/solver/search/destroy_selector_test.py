from numpy.random import default_rng

from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.search.destroy_selector import (
    AlnsConfig,
    DestroyKind,
    DestroySelector,
)
from baseline_core.core.optimization.solver.search.kernel_state import KernelState
from baseline_core.core.optimization.solver.search.search_index import SearchIndex


class TestTail:
    def test__a_tail__is_what_follows_a_cut_of_one_machine_its_first_operation_kept(
        self, shop: BoundModel
    ) -> None:
        """Four operations in a row on M1: a tail is a suffix of that row, never all of it."""
        shop.kernel.load(start=[0, 3, 6, 9], machine=[0, 0, 0, 0])
        selector = DestroySelector(SearchIndex.of(shop), AlnsConfig(), default_rng(0), 0)
        selector.weights = [
            1.0 if arm.kind == DestroyKind.TAIL and arm.cap == 1 else 0.0 for arm in selector.ARMS
        ]

        moves = [selector.select(KernelState(shop.kernel)) for _ in range(20)]

        suffixes = [[1, 2, 3], [2, 3], [3]]
        assert all(move is not None and move.indices in suffixes for move in moves)
        assert all(move is not None and move.in_order and move.remove_first for move in moves)

    def test__a_tail__is_nothing_when_no_set_holds_two_elements(self, shop: BoundModel) -> None:
        """One operation per machine leaves nothing to cut."""
        shop.kernel.load(start=[0, 0, 4, 4], machine=[0, 1, 0, 1])
        shop.kernel.remove(2)
        shop.kernel.remove(3)
        selector = DestroySelector(SearchIndex.of(shop), AlnsConfig(), default_rng(0), 0)
        selector.weights = [1.0 if arm.kind == DestroyKind.TAIL else 0.0 for arm in selector.ARMS]

        move = selector.select(KernelState(shop.kernel))

        assert move is None
