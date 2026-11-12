"""Collections a bound model ranges over: known subsets, decided sets, subsets read at an index."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from baseline_core.core.optimization.problem_formulation.declaration import Collection
from baseline_core.core.optimization.problem_formulation.expression import Expr


class FixedSubset(Collection[Any]):
    """A subset of the elements known once the model is bound, in a fixed order."""

    def __init__(self, elements: list[int], *, decision: bool = True) -> None:
        """Keep the elements.

        Args:
            elements: The element positions.
            decision: Whether they belong to the decision set (absent elements are then skipped).
        """
        self.elements = [int(element) for element in elements]
        self.decision = decision


class PartitionPart(Collection[Any]):
    """The elements the search assigns to one target."""

    def __init__(self, index: int) -> None:
        """Keep the target.

        Args:
            index: The target's position.
        """
        self.index = index


class ComputedSubset(Collection[Any]):
    """The subset ``flat[offsets[key]:offsets[key + 1]]`` of a family read at a computed index."""

    def __init__(self, flat: NDArray[np.int64], offsets: NDArray[np.int64], key: Expr) -> None:
        """Keep the flattened family and the index expression.

        Args:
            flat: The members of every subset, one after another.
            offsets: Where each subset starts in ``flat``, plus the end.
            key: Which subset, per instance.
        """
        self.flat = flat
        self.offsets = offsets
        self.key = key


class BoundFilteredSet(Collection[Any]):
    """A bound collection restricted by a mask: one flag per element, non-zero to keep it."""

    def __init__(self, base: Collection[Any], mask: NDArray[Any]) -> None:
        """Keep the base collection and the mask's data.

        Args:
            base: The collection filtered, bound.
            mask: One flag per element.

        Raises:
            ValueError: The mask is not one flat array of flags.
        """
        if not isinstance(mask, np.ndarray) or mask.ndim != 1:
            msg = "a bound filter's mask is one flat array, one flag per element"
            raise ValueError(msg)

        self.base = base
        self.mask = mask
