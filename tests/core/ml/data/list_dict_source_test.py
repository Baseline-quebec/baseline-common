"""Tests for the data source over a list of dicts."""

from typing import Any, ClassVar

from baseline_core.core.ml.data.list_dict_source import ListDictSource


class TestListDictSource:
    """Tests for ``ListDictSource``."""

    EXPECTED_LENGTH: ClassVar[int] = 3
    EXPECTED_SECOND_FEATURE1: ClassVar[int] = 15

    def test__feature_names__should_return_keys_of_first_record(
        self, list_source: ListDictSource
    ) -> None:
        """Keys of the first record are the feature names."""
        names = list_source.field_names

        assert names == ["feature1", "feature2", "feature3"]

    def test__feature_names__when_empty__should_return_empty_list(
        self, empty_source: ListDictSource
    ) -> None:
        """No records, no feature names."""
        assert empty_source.field_names == []
        assert len(empty_source) == 0

    def test__len__should_return_record_count(self, list_source: ListDictSource) -> None:
        """Length is the number of records."""
        assert len(list_source) == self.EXPECTED_LENGTH

    def test__get_value__should_return_record_key(
        self, list_source: ListDictSource, records: list[dict[str, Any]]
    ) -> None:
        """A value is addressed by record index and key."""
        value = list_source.get_value(1, "feature1")

        assert value == self.EXPECTED_SECOND_FEATURE1
        assert value == records[1]["feature1"]
