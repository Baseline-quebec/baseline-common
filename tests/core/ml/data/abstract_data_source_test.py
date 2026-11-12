"""Tests for the behavior every data source shares."""

from typing import Any

import pytest

from baseline_core.core.ml.data.list_dict_source import ListDictSource
from tests.core.ml.dummy_classes import UpperCaseLoader


class TestAbstractDataSource:
    """Tests for the behavior every source shares, through ``ListDictSource``."""

    def test__select__should_keep_the_chosen_samples_and_the_loaders(
        self, loaded_source: ListDictSource
    ) -> None:
        """A phase's source has its samples, in the given order, still loaded when read."""
        selected = loaded_source.select([2, 0])

        assert isinstance(selected, ListDictSource)
        assert [selected.get_value(index, "feature3") for index in range(2)] == ["TEST", "HELLO"]

    def test__get_value__when_feature_has_a_loader__should_return_the_loaded_value(
        self, loaded_source: ListDictSource, records: list[dict[str, Any]]
    ) -> None:
        """A feature stored as a reference comes back loaded; the stored value is kept."""
        assert loaded_source.get_value(0, "feature3") == "HELLO"
        assert loaded_source.get_stored_value(0, "feature3") == "hello"
        assert loaded_source.get_value(0, "feature1") == records[0]["feature1"]

    def test__init__when_loader_names_no_feature__should_raise_value_error(
        self, records: list[dict[str, Any]]
    ) -> None:
        """A loader for a feature the data lacks is a configuration mistake."""
        with pytest.raises(ValueError, match=r"unknown features \['ghost'\]"):
            ListDictSource(records, loaders={"ghost": UpperCaseLoader()})
