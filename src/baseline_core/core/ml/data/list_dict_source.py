"""Data source over a list of dicts."""

from collections.abc import Sequence
from typing import Any

from baseline_core.core.ml.data.abstract_data_source import AbstractDataSource


class ListDictSource(AbstractDataSource[list[dict[str, Any]]]):
    """Records of a list of dicts (decoded JSON lines, API responses, ...)."""

    @property
    def field_names(self) -> list[str]:
        """Keys of the first record; empty when there is no data."""
        return list(self.data[0]) if self.data else []

    def __len__(self) -> int:
        """Number of records."""
        return len(self.data)

    def get_stored_value(self, index: int, key: str) -> Any:  # noqa: ANN401
        """Return ``key`` of record ``index``."""
        return self.data[index][key]

    def select_data(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Return the records at ``indices``."""
        return [self.data[index] for index in indices]
