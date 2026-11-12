import numpy as np
import pytest

from baseline_core.core.optimization.solver.binding.bound_collection import (
    BoundFilteredSet,
    FixedSubset,
)


class TestBoundFilteredSet:
    def test__a_mask_that_is_no_flat_array__is_refused_when_the_filter_is_made(self) -> None:
        """A list, or an array of two dimensions, is refused at construction, not at reading."""
        base = FixedSubset([0, 1])

        with pytest.raises(ValueError, match="mask"):
            BoundFilteredSet(base, [1, 0])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="mask"):
            BoundFilteredSet(base, np.ones((2, 1), dtype=np.int64))

    def test__a_flat_mask__is_kept_as_given(self) -> None:
        """A flat array of flags is the filter's mask."""
        mask = np.array([1, 0], dtype=np.int64)
        assert BoundFilteredSet(FixedSubset([0, 1]), mask).mask is mask
