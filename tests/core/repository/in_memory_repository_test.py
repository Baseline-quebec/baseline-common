"""Unit tests for the in-memory repository: filters applied in Python, and its writes."""

import pytest

from baseline_core.core.repository.abstract_repository import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_core.core.repository.in_memory_repository import InMemoryRepository
from baseline_core.core.repository.query_filters import (
    ComparisonFilter,
    EqualityFilter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)
from tests.core.repository.conftest import Equipment


class TestFilters:
    """Every filter holds or not on a record's fields, text regardless of case."""

    async def test__an_equality__keeps_the_matching_records(
        self, inventory: list[Equipment]
    ) -> None:
        """The active pieces, the retired one left out."""
        repository = InMemoryRepository(Equipment, inventory)

        found = await repository.get_by_filters(
            [EqualityFilter(field="status", value="active")], limit=None
        )

        assert [piece.id_ for piece in found] == ["m-1", "m-2", "m-4"]

    async def test__text__compares_regardless_of_case(self, inventory: list[Equipment]) -> None:
        """A name typed in lower case still matches, on a nested field too."""
        repository = InMemoryRepository(Equipment, inventory)

        by_name = await repository.get_by_filters([
            EqualityFilter(field="name", value="drill bravo")
        ])
        by_city = await repository.get_by_filters([
            TextFilter(field="site.city", value="RIVIÈRES")
        ])

        assert [piece.id_ for piece in by_name] == ["m-2"]
        assert [piece.id_ for piece in by_city] == ["m-4"]

    async def test__a_logical_or__keeps_records_any_operand_holds_for(
        self, inventory: list[Equipment]
    ) -> None:
        """Either brand."""
        repository = InMemoryRepository(Equipment, inventory)
        brands = LogicalFilter(
            operator="|",
            operands=[
                EqualityFilter(field="brand", value="Mazak"),
                EqualityFilter(field="brand", value="Okuma"),
            ],
        )

        found = await repository.get_by_filters([brands])

        assert [piece.id_ for piece in found] == ["m-2", "m-3"]

    async def test__a_missing_value__is_only_different(self, inventory: list[Equipment]) -> None:
        """No bay equals nothing, and differs from anything."""
        repository = InMemoryRepository(Equipment, inventory)

        unequal = await repository.get_by_filters([
            ComparisonFilter(operator="!=", field="bay", value="C9")
        ])
        equal = await repository.get_by_filters([EqualityFilter(field="bay", value="C9")])

        assert len(unequal) == len(inventory)
        assert equal == []


class TestPaging:
    """Sorted, then sliced by offset and limit."""

    async def test__sorted_descending__then_sliced(self, inventory: list[Equipment]) -> None:
        """The second and third names, from the last."""
        repository = InMemoryRepository(Equipment, inventory)

        found = await repository.get_by_filters(
            sort_by="name", sort_order=SortOrder.DESC, limit=2, offset=1
        )

        assert [piece.name for piece in found] == ["PRESS ALPHA", "LATHE DELTA"]

    async def test__missing_values__sort_last(self, inventory: list[Equipment]) -> None:
        """Only the press has a bay."""
        repository = InMemoryRepository(Equipment, inventory)

        found = await repository.get_by_filters(sort_by="bay", limit=None)

        assert found[0].id_ == "m-1"


class TestGetById:
    """A record found by its id field."""

    async def test__the_primary_key__finds_the_record(self, inventory: list[Equipment]) -> None:
        """A piece's id is the field named as its primary key."""
        repository = InMemoryRepository(Equipment, inventory, primary_key="id_")

        assert await repository.get_by_id("m-4") == inventory[3]
        assert await repository.get_by_id("m-9") is None


class TestWrites:
    """Records inserted, replaced and deleted by their id."""

    async def test__an_inserted_record__is_found(self, press: Equipment, lathe: Equipment) -> None:
        """Insert returns the id the record is then found by."""
        repository = InMemoryRepository(Equipment, [press], primary_key="id_")

        id_ = await repository.insert_one(lathe)

        assert id_ == "m-4"
        assert await repository.get_by_id("m-4") == lathe

    async def test__a_held_id__is_not_inserted_twice(self, press: Equipment) -> None:
        """Nothing is added when one record of the batch is already held."""
        repository = InMemoryRepository(Equipment, [press], primary_key="id_")

        with pytest.raises(DuplicateRecordError, match="m-1"):
            await repository.insert_many([press])

        assert len(await repository.get_by_filters()) == 1

    async def test__an_update__replaces_the_record_of_the_same_id(self, press: Equipment) -> None:
        """The record, as it now is."""
        repository = InMemoryRepository(Equipment, [press], primary_key="id_")
        closed = press.model_copy(update={"status": "retired"})

        await repository.update_one(closed)

        assert await repository.get_by_id("m-1") == closed

    async def test__a_record_not_held__cannot_be_updated_nor_deleted(
        self, press: Equipment, lathe: Equipment
    ) -> None:
        """Both name the missing id."""
        repository = InMemoryRepository(Equipment, [press], primary_key="id_")

        with pytest.raises(RecordNotFoundError, match="m-4"):
            await repository.update_one(lathe)

        with pytest.raises(RecordNotFoundError, match="m-4"):
            await repository.delete_one("m-4")

    async def test__a_delete_of_many__removes_none_when_one_is_not_held(
        self, press: Equipment, lathe: Equipment
    ) -> None:
        """The missing id is named, and every record stays."""
        repository = InMemoryRepository(Equipment, [press, lathe], primary_key="id_")

        with pytest.raises(RecordNotFoundError, match="m-9"):
            await repository.delete_many(["m-1", "m-9"])

        assert len(await repository.get_by_filters(limit=None)) == 2

    async def test__a_deleted_record__is_gone(self, press: Equipment) -> None:
        """Delete returns the id removed."""
        repository = InMemoryRepository(Equipment, [press], primary_key="id_")

        id_ = await repository.delete_one("m-1")

        assert id_ == "m-1"
        assert await repository.get_by_id("m-1") is None

    async def test__an_upsert__replaces_every_record_of_the_id_and_adds_the_others(
        self, press: Equipment, lathe: Equipment
    ) -> None:
        """Two held under one id, as a dirty file may; one left, the last given; one added."""
        twin = press.model_copy(update={"name": "TWIN"})
        repository = InMemoryRepository(Equipment, [press, twin], primary_key="id_")
        renamed = press.model_copy(update={"name": "RENAMED"})
        again = press.model_copy(update={"name": "AGAIN"})

        ids = await repository.upsert_many([renamed, lathe, again])

        assert ids == ["m-1", "m-4", "m-1"]
        assert await repository.get_by_filters(limit=None) == [again, lathe]

    async def test__a_record_without_an_id__is_given_one(self, press: Equipment) -> None:
        """As a backend would: the id returned finds the record, the id set."""
        repository = InMemoryRepository(Equipment, [press], primary_key="id_")
        draft = press.model_copy(update={"id_": None, "serial": "S-009"})

        first, second = await repository.insert_many([draft, draft])

        assert first != second
        held = await repository.get_by_id(first)
        assert held is not None
        assert (held.id_, held.serial) == (first, "S-009")
