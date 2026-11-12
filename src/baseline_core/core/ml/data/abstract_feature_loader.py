"""Loaders turning the stored value of a feature into the data it points to."""

from abc import ABC, abstractmethod


class AbstractFeatureLoader[Stored, Loaded](ABC):
    """Turns the stored value of a feature (a path, a URI, ...) into the data it points to."""

    @abstractmethod
    def load(self, stored: Stored) -> Loaded:
        """Load the data ``stored`` points to.

        Args:
            stored: The value kept in the data source.

        Returns:
            The loaded data.
        """
