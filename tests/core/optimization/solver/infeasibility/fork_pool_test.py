import operator
from functools import partial

import pytest

from baseline_core.core.optimization.solver.infeasibility.fork_pool import ForkPool


class TestForkPool:
    def test__results__come_back_in_the_items_order(self) -> None:
        """Items shared out between copies come back where they were, each computed once."""
        assert ForkPool(3).map(abs, list(range(-5, 5))) == [abs(n) for n in range(-5, 5)]

    def test__each_item__runs_in_a_forked_copy(self) -> None:
        """With several workers the items change the state of the copies, not this process."""
        seen: list[int] = []
        ForkPool(2).map(seen.append, [0, 1, 2, 3])

        assert seen == []

    def test__a_failure_in_a_copy__is_raised_here_with_its_traceback(self) -> None:
        """An error in a forked copy reaches the caller, with what happened there."""
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            ForkPool(2).map(partial(operator.floordiv, 1), [1, 0, 2])

    def test__one_worker__runs_here(self) -> None:
        """One worker computes the items in this process."""
        seen: list[int] = []
        ForkPool(1).map(seen.append, [0, 1])

        assert seen == [0, 1]
