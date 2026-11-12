"""Tests for ``DataFrameSource``."""

from typing import ClassVar

from baseline_core.core.ml.modality.tabular.data_frame_source import DataFrameSource


class TestDataFrameSource:
    """Tests for ``DataFrameSource``."""

    EXPECTED_LENGTH: ClassVar[int] = 3
    EXPECTED_SECOND_AGE: ClassVar[int] = 30

    def test__feature_names__should_return_columns(
        self, dataframe_source: DataFrameSource
    ) -> None:
        """Column names are the feature names."""
        names = dataframe_source.field_names

        assert names == ["age", "name", "score"]

    def test__len__should_return_row_count(self, dataframe_source: DataFrameSource) -> None:
        """Length is the number of rows."""
        assert len(dataframe_source) == self.EXPECTED_LENGTH

    def test__get_value__should_return_cell(self, dataframe_source: DataFrameSource) -> None:
        """A value is addressed by row index and column name."""
        age = dataframe_source.get_value(1, "age")
        name = dataframe_source.get_value(1, "name")

        assert age == self.EXPECTED_SECOND_AGE
        assert name == "Bob"

    def test__select__should_renumber_the_rows(self, dataframe_source: DataFrameSource) -> None:
        """Selected rows are addressed from 0, like any source."""
        selected = dataframe_source.select([2])

        assert len(selected) == 1
        assert selected.get_value(0, "name") == "Charlie"

    def test__get_value__when_column_has_a_loader__should_return_the_loaded_value(
        self, loaded_dataframe_source: DataFrameSource
    ) -> None:
        """Loading does not depend on the container."""
        assert loaded_dataframe_source.get_value(1, "name") == "BOB"
