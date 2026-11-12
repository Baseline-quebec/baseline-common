"""The contract a score fulfills to be frozen into a baseline and gated against it."""

from abc import ABC, abstractmethod
from collections.abc import Mapping


class AbstractGatedScore(ABC):
    """The contract a score wears to be frozen into a baseline and compared against it."""

    @abstractmethod
    def gated_numbers(self) -> Mapping[str, float]:
        """Name every number of this score a gate may refer to.

        Returns:
            The value of each number, keyed by the name the gate's fields use.
        """
