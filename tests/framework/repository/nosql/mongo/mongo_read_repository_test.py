"""Unit tests for the MongoDB reader: reads by id, filters, sorting and pages."""

from typing import Any, Literal, cast

import pytest
from pydantic import BaseModel
from pymongo.collection import Collection

from baseline_common.framework.repository.nosql.mongo.mongo_read_repository import (
    MongoReadRepository,
)
from baseline_common.framework.repository.query_filters import (
    ComparisonFilter,
    EqualityFilter,
    Filter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)
from tests.framework.repository.nosql.mongo.async_mongomock import AsyncMongomock
from tests.framework.repository.nosql.mongo.machine import Machine


class Prefix(Filter):
    """A condition MongoDB is not taught: a field starting with a value."""

    operator: Literal["in"] = "in"
    """Always containment, so the filter is a valid ``Filter``."""
    field: str
    """The field read."""
    value: str
    """The value the field starts with."""

    def matches(self, record: BaseModel) -> bool:
        """Tell whether the field starts with the value.

        Args:
            record: The record.

        Returns:
            Whether it holds.
        """
        return str(self.value_of(record, self.field)).startswith(self.value)


class Counted(BaseModel):
    """A record whose id is a number, which a read by text id would never find."""

    id: int
    """A numeric id."""


class TestConstruction:
    """The primary key is checked once, when the reader is made."""

    @staticmethod
    def test__when_a_primary_key_is_not_a_field__raises_value_error(
        stored: Collection[dict[str, Any]],
    ) -> None:
        """No document could be named."""
        collection = cast("Any", AsyncMongomock(stored))

        with pytest.raises(ValueError, match="'serial'"):
            MongoReadRepository(Machine, collection, primary_key="serial")

    @staticmethod
    def test__when_a_primary_key_is_not_text__raises_value_error(
        stored: Collection[dict[str, Any]],
    ) -> None:
        """get_by_id names a record by text."""
        collection = cast("Any", AsyncMongomock(stored))

        with pytest.raises(ValueError, match="Counted"):
            MongoReadRepository(Counted, collection)


class TestGetById:
    """A read by id goes to the _id index."""

    @staticmethod
    async def test__when_an_id_is_held__returns_its_record(
        stocked: MongoReadRepository[Machine], fleet: list[Machine]
    ) -> None:
        """The record, as inserted."""
        found = await stocked.get_by_id("m-3")

        assert found == fleet[2]

    @staticmethod
    async def test__when_an_id_is_not_held__returns_none(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """None, not an error."""
        found = await stocked.get_by_id("m-9")

        assert found is None

    @staticmethod
    @pytest.mark.usefixtures("stocked")
    async def test__when_a_record_is_held__returns_its_id_as_the_document_id_only(
        stored: Collection[dict[str, Any]],
    ) -> None:
        """Once, as _id, not again as id_."""
        document = stored.find_one({"_id": "m-1"})

        assert document is not None
        assert "id_" not in document


class TestFilters:
    """Every filter becomes a MongoDB query that keeps the same records."""

    @staticmethod
    async def test__when_an_equality_is_given__returns_the_matching_records(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """The three lathes."""
        found = await stocked.get_by_filters([EqualityFilter(field="kind", value="lathe")])

        assert [machine.id_ for machine in found] == ["m-1", "m-2", "m-5"]

    @staticmethod
    async def test__when_an_equality_is_on_text__returns_no_match_of_another_case(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """Unlike the in-memory repository: case-insensitive, no index would serve it."""
        found = await stocked.get_by_filters([EqualityFilter(field="kind", value="LATHE")])

        assert found == []

    @staticmethod
    async def test__when_a_difference_is_given__returns_the_others(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """The press and the mill."""
        found = await stocked.get_by_filters([
            ComparisonFilter(operator="!=", field="kind", value="lathe")
        ])

        assert [machine.id_ for machine in found] == ["m-3", "m-4"]

    @staticmethod
    async def test__when_a_record_lacks_the_field__returns_it_for_a_difference(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """The mill was never serviced; as in memory, a missing value only differs."""
        found = await stocked.get_by_filters([
            ComparisonFilter(operator="!=", field="serviced", value="2026-09-01T00:00:00Z")
        ])

        assert [machine.id_ for machine in found] == ["m-2", "m-3", "m-4", "m-5"]

    @staticmethod
    async def test__when_a_bound_is_on_a_number__returns_the_records_within_it(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """At least 7.25 hours."""
        found = await stocked.get_by_filters([
            ComparisonFilter(operator=">=", field="hours", value=7.25)
        ])

        assert [machine.id_ for machine in found] == ["m-1", "m-3", "m-4"]

    @staticmethod
    async def test__when_a_bound_is_on_a_datetime__returns_the_records_within_its_iso_text(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """Serviced after the 10th, written as a filter writes a date."""
        found = await stocked.get_by_filters([
            ComparisonFilter(operator=">", field="serviced", value="2026-09-10")
        ])

        assert [machine.id_ for machine in found] == ["m-2", "m-3"]

    @staticmethod
    async def test__when_a_path_is_nested__returns_the_matches_of_the_nested_field(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """The machines of Montréal."""
        found = await stocked.get_by_filters([EqualityFilter(field="site.city", value="Montréal")])

        assert [machine.id_ for machine in found] == ["m-1", "m-2"]

    @staticmethod
    async def test__when_an_equality_is_on_a_list__returns_records_holding_the_element(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """MongoDB matches any element of an array."""
        found = await stocked.get_by_filters([EqualityFilter(field="tags", value="south")])

        assert [machine.id_ for machine in found] == ["m-1", "m-3"]

    @staticmethod
    async def test__when_a_filter_is_on_the_primary_key__returns_matches_on_the_document_id(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """id_ is stored as _id."""
        found = await stocked.get_by_filters([
            ComparisonFilter(operator=">", field="id_", value="m-3")
        ])

        assert [machine.id_ for machine in found] == ["m-4", "m-5"]

    @staticmethod
    async def test__when_an_either_or_is_given__returns_the_matches_of_either(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """The press or the mill."""
        found = await stocked.get_by_filters([
            LogicalFilter(
                operator="|",
                operands=[
                    EqualityFilter(field="kind", value="press"),
                    EqualityFilter(field="kind", value="mill"),
                ],
            )
        ])

        assert [machine.id_ for machine in found] == ["m-3", "m-4"]

    @staticmethod
    async def test__when_a_conjunction_is_given__returns_the_matches_of_both(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """A lathe of Montréal on two shifts."""
        found = await stocked.get_by_filters([
            LogicalFilter(
                operator="&",
                operands=[
                    EqualityFilter(field="site.city", value="Montréal"),
                    EqualityFilter(field="shifts", value=2),
                ],
            )
        ])

        assert [machine.id_ for machine in found] == ["m-1"]

    @staticmethod
    async def test__when_operands_are_empty__returns_all_for_an_and_none_for_an_or(
        stocked: MongoReadRepository[Machine], fleet: list[Machine]
    ) -> None:
        """As all() and any() of nothing; MongoDB refuses an empty $and or $or."""
        every = await stocked.get_by_filters([LogicalFilter(operator="&", operands=[])])
        none = await stocked.get_by_filters([LogicalFilter(operator="|", operands=[])])

        assert len(every) == len(fleet)
        assert none == []

    @staticmethod
    async def test__when_a_text_filter_is_given__returns_partial_matches_regardless_of_case(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """Its hyphen is a character, not a regex range."""
        found = await stocked.get_by_filters([TextFilter(field="site.city", value="trois-riv")])

        assert [machine.id_ for machine in found] == ["m-3"]

    @staticmethod
    async def test__when_a_text_filter_has_a_dot__returns_no_match_of_another_character(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """A dot matches only a dot."""
        found = await stocked.get_by_filters([TextFilter(field="name", value="Lathe.")])

        assert found == []

    @staticmethod
    async def test__when_a_filter_is_unknown__raises_value_error(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """Never ignored, which would return too much."""
        with pytest.raises(ValueError, match="Prefix"):
            await stocked.get_by_filters([Prefix(field="name", value="Lathe")])

    @staticmethod
    async def test__when_a_count_is_asked__returns_the_number_of_matches(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """Three lathes, five machines."""
        lathes = await stocked.count_by_filters([EqualityFilter(field="kind", value="lathe")])
        machines = await stocked.count_by_filters([])

        assert [lathes, machines] == [3, 5]


class TestSortAndPage:
    """Results are sorted, ties broken by id, and paged."""

    @staticmethod
    async def test__when_no_sort_is_given__returns_records_by_id(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """Every machine, by id."""
        found = await stocked.get_by_filters()

        assert [machine.id_ for machine in found] == ["m-1", "m-2", "m-3", "m-4", "m-5"]

    @staticmethod
    async def test__when_a_sort_has_ties__returns_the_tied_by_id(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """Most hours first; m-1 and m-3 tie at 12.5."""
        found = await stocked.get_by_filters(sort_by="hours", sort_order=SortOrder.DESC)

        assert [machine.id_ for machine in found] == ["m-1", "m-3", "m-4", "m-2", "m-5"]

    @staticmethod
    async def test__when_a_value_is_missing__returns_it_as_the_lowest(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """The mill, never serviced, first in ascending order."""
        found = await stocked.get_by_filters(sort_by="serviced")

        assert [machine.id_ for machine in found] == ["m-4", "m-1", "m-5", "m-2", "m-3"]

    @staticmethod
    async def test__when_a_page_is_asked__returns_the_limit_past_the_offset(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """The second page of two."""
        found = await stocked.get_by_filters(sort_by="shifts", limit=2, offset=2)

        assert [machine.id_ for machine in found] == ["m-1", "m-3"]

    @staticmethod
    async def test__when_no_limit_is_given__returns_every_match(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """After the offset."""
        found = await stocked.get_by_filters(limit=None, offset=1)

        assert len(found) == 4

    @staticmethod
    async def test__when_a_limit_is_zero__returns_nothing(
        stocked: MongoReadRepository[Machine],
    ) -> None:
        """MongoDB reads a zero limit as none."""
        found = await stocked.get_by_filters(limit=0)

        assert found == []
