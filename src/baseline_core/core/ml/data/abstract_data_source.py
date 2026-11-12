"""Adapters giving uniform row/key access to raw data containers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Self

from baseline_core.core.ml.data.abstract_feature_loader import AbstractFeatureLoader


class AbstractDataSource[Container](ABC):
    """Uniform access to the samples and keys of a raw data container.

    Attributes:
        data: The wrapped container.
        loaders: Loader of every feature stored as a reference, keyed by feature name.
    """

    def __init__(
        self, data: Container, loaders: Mapping[str, AbstractFeatureLoader[Any, Any]] | None = None
    ) -> None:
        """Wrap ``data`` and check that every loader names a feature.

        Args:
            data: The raw data container.
            loaders: Loader of every feature stored as a reference, keyed by feature name.

        Raises:
            ValueError: If a loader names no feature of ``data``.
        """
        self.data = data
        self.loaders = dict(loaders or {})
        unknown = sorted(set(self.loaders) - set(self.field_names))

        if unknown:
            error_msg = f"Loaders for unknown features {unknown}; features: {self.field_names}"
            raise ValueError(error_msg)

    @property
    @abstractmethod
    def field_names(self) -> list[str]:
        """Keys available on every sample."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples."""

    @abstractmethod
    def get_stored_value(self, index: int, key: str) -> Any:  # noqa: ANN401
        """Return the value of ``key`` for sample ``index`` as the container holds it."""

    @abstractmethod
    def select_data(self, indices: Sequence[int]) -> Container:
        """Return the container restricted to the samples at ``indices``, in that order."""

    def get_value(self, index: int, key: str) -> Any:  # noqa: ANN401
        """Return the value of ``key`` for sample ``index``, loaded if ``key`` has a loader."""
        stored = self.get_stored_value(index, key)
        loader = self.loaders.get(key)
        return stored if loader is None else loader.load(stored)

    def select(self, indices: Sequence[int]) -> Self:
        """Return a source over the samples at ``indices`` only, with the same loaders.

        Args:
            indices: Sample indices, e.g. the ones of one phase.

        Returns:
            The smaller source.
        """
        return type(self)(self.select_data(indices), self.loaders)
