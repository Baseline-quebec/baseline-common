"""Fixtures of the tabular modality tests: a DataFrame and its sources."""

import pandas as pd
import pytest

from baseline_core.core.ml.modality.tabular.data_frame_source import DataFrameSource
from tests.core.ml.dummy_classes import UpperCaseLoader


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Three rows with mixed column types."""
    return pd.DataFrame({
        "age": [25, 30, 35],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85, 90, 95],
    })


@pytest.fixture
def dataframe_source(dataframe: pd.DataFrame) -> DataFrameSource:
    """Source over the DataFrame."""
    return DataFrameSource(dataframe)


@pytest.fixture
def loaded_dataframe_source(dataframe: pd.DataFrame) -> DataFrameSource:
    """Source over the DataFrame whose ``name`` is loaded when read."""
    return DataFrameSource(dataframe, loaders={"name": UpperCaseLoader()})
