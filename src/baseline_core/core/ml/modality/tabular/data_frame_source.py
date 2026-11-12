"""Tables as data sources: one sample per row."""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from baseline_core.core.ml.data.abstract_data_source import AbstractDataSource


class DataFrameSource(AbstractDataSource[pd.DataFrame]):
    """Rows of a pandas DataFrame."""

    @property
    def field_names(self) -> list[str]:
        """Column names."""
        return [str(column) for column in self.data.columns]

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.data)

    def get_stored_value(self, index: int, key: str) -> Any:  # noqa: ANN401
        """Return the cell at row ``index``, column ``key``."""
        return self.data.iloc[index][key]

    def select_data(self, indices: Sequence[int]) -> pd.DataFrame:
        """Return the rows at ``indices``, renumbered from 0."""
        return self.data.iloc[list(indices)].reset_index(drop=True)
