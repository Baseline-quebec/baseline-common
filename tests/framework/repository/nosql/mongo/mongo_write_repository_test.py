"""Unit tests for the MongoDB writer: the write contract, all or nothing where it can be."""

import re
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pymongo.collection import Collection

from baseline_common.framework.repository.abstract_repository import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_common.framework.repository.nosql.mongo.mongo_read_repository import (
    MongoReadRepository,
)
from baseline_common.framework.repository.nosql.mongo.mongo_write_repository import (
    MongoWriteRepository,
)
from baseline_common.framework.repository.query_filters import EqualityFilter
from tests.framework.repository.equipment import Site
from tests.framework.repository.nosql.mongo.machine import Machine


class TestInsert:
    """An insert adds new records and refuses an id already held."""

    @staticmethod
    async def test__when_many_records_are_given__are_inserted_in_order(
        machine_reads: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Their ids, as given, and each read back whole."""
        inserted = await machine_writes.insert_many(fleet)

        assert inserted == ["m-1", "m-2", "m-3", "m-4", "m-5"]
        assert await machine_reads.get_by_filters() == fleet

    @staticmethod
    async def test__when_a_record_has_no_id__is_given_an_object_id(
        machine_reads: MongoReadRepository[Machine], machine_writes: MongoWriteRepository[Machine]
    ) -> None:
        """Text of 24 hex digits, the record read back under it."""
        inserted = await machine_writes.insert_one(Machine(name="Drill", site=Site(city="Laval")))

        assert re.fullmatch(r"[0-9a-f]{24}", inserted)
        assert await machine_reads.get_by_id(inserted) == Machine(
            id_=inserted, name="Drill", site=Site(city="Laval")
        )

    @staticmethod
    async def test__when_nothing_is_inserted__writes_nothing(
        machine_writes: MongoWriteRepository[Machine],
    ) -> None:
        """An empty insert, which pymongo refuses, is never sent."""
        inserted = await machine_writes.insert_many([])

        assert inserted == []

    @staticmethod
    async def test__when_an_id_is_held__is_refused_before_any_write(
        machine_reads: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """An insert never replaces; an upsert does."""
        await machine_writes.insert_one(fleet[0])

        with pytest.raises(DuplicateRecordError, match="m-1"):
            await machine_writes.insert_many([fleet[1], fleet[0]])

        assert await machine_reads.count_by_filters([]) == 1

    @staticmethod
    async def test__when_an_id_is_given_twice__is_refused_before_any_write(
        machine_reads: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Nothing of the batch is written."""
        with pytest.raises(DuplicateRecordError, match="m-2"):
            await machine_writes.insert_many([fleet[1], fleet[0], fleet[1]])

        assert await machine_reads.count_by_filters([]) == 0

    @staticmethod
    async def test__when_an_id_is_taken_since_the_check__is_refused_by_the_index(
        machine_writes: MongoWriteRepository[Machine],
        stored: Collection[dict[str, Any]],
        fleet: list[Machine],
    ) -> None:
        """Another writer took m-2: the unique _id refuses it, after m-1 is written."""
        stored.insert_one({"_id": "m-2", "name": "Lathe B"})

        with (
            patch.object(machine_writes, "_held", AsyncMock(return_value=set())),
            pytest.raises(DuplicateRecordError, match="m-2"),
        ):
            await machine_writes.insert_many(fleet[:3])

        assert sorted(document["_id"] for document in stored.find()) == ["m-1", "m-2"]

    @staticmethod
    @pytest.mark.usefixtures("stocked")
    async def test__when_a_record_has_a_datetime__is_stored_as_iso_text(
        stored: Collection[dict[str, Any]],
    ) -> None:
        """As a filter writes it, so the two compare."""
        document = stored.find_one({"_id": "m-1"})

        assert document is not None
        assert document["serviced"] == "2026-09-01T00:00:00Z"


class TestUpdate:
    """An update replaces records held, and never inserts."""

    @staticmethod
    async def test__when_a_record_is_updated__replaces_the_record(
        stocked: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The new hours are read back."""
        updated = await machine_writes.update_one(fleet[0].model_copy(update={"hours": 20.0}))

        found = await stocked.get_by_id("m-1")

        assert updated == "m-1"
        assert found is not None
        assert found.hours == pytest.approx(20.0)

    @staticmethod
    async def test__when_a_record_is_not_held__is_refused_before_any_write(
        stocked: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The first lathe keeps its hours when the batch names an unknown machine."""
        with pytest.raises(RecordNotFoundError, match="m-9"):
            await machine_writes.update_many([
                fleet[0].model_copy(update={"hours": 20.0}),
                Machine(id_="m-9", name="Drill", site=Site(city="Laval")),
            ])

        assert await stocked.get_by_id("m-1") == fleet[0]
        assert await stocked.get_by_id("m-9") is None

    @staticmethod
    async def test__when_nothing_is_updated__updates_nothing(
        machine_writes: MongoWriteRepository[Machine],
    ) -> None:
        """An empty bulk, which pymongo refuses, is never sent."""
        updated = await machine_writes.update_many([])

        assert updated == []


class TestUpsert:
    """An upsert inserts or replaces."""

    @staticmethod
    async def test__when_records_are_upserted__inserts_or_replaces(
        stocked: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """A new drill, and the first lathe renamed; the last of an id repeated is kept."""
        drill = Machine(id_="m-9", name="Drill", site=Site(city="Laval"))
        renamed = fleet[0].model_copy(update={"name": "Lathe Z"})

        upserted = await machine_writes.upsert_many([drill, fleet[0], renamed])

        assert upserted == ["m-9", "m-1", "m-1"]
        assert await stocked.get_by_id("m-9") == drill
        assert await stocked.get_by_id("m-1") == renamed
        assert await stocked.count_by_filters([]) == len(fleet) + 1

    @staticmethod
    async def test__when_a_record_has_no_id__is_inserted(
        machine_reads: MongoReadRepository[Machine], machine_writes: MongoWriteRepository[Machine]
    ) -> None:
        """Under a fresh id."""
        upserted = await machine_writes.upsert_one(Machine(name="Drill", site=Site(city="Laval")))

        found = await machine_reads.get_by_id(upserted)

        assert found is not None
        assert found.name == "Drill"


class TestDelete:
    """A delete removes records by id or by filters, and refuses what would surprise."""

    @staticmethod
    async def test__when_an_id_is_held__deletes_its_record(
        stocked: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The others stay."""
        deleted = await machine_writes.delete_one("m-1")

        assert deleted == "m-1"
        assert await stocked.get_by_id("m-1") is None
        assert await stocked.count_by_filters([]) == len(fleet) - 1

    @staticmethod
    @pytest.mark.usefixtures("stocked")
    async def test__when_an_id_is_not_held__is_refused(
        machine_writes: MongoWriteRepository[Machine],
    ) -> None:
        """Not a silent no-op."""
        with pytest.raises(RecordNotFoundError, match="m-9"):
            await machine_writes.delete_one("m-9")

    @staticmethod
    async def test__when_one_of_the_ids_is_not_held__deletes_none(
        stocked: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The missing id is named, and every record stays."""
        with pytest.raises(RecordNotFoundError, match="m-9"):
            await machine_writes.delete_many(["m-1", "m-9"])

        assert await stocked.count_by_filters([]) == len(fleet)

    @staticmethod
    async def test__when_ids_are_held__delete_their_records_in_one_call(
        stocked: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The others stay."""
        deleted = await machine_writes.delete_many(["m-1", "m-2"])

        assert deleted == ["m-1", "m-2"]
        assert await stocked.count_by_filters([]) == len(fleet) - 2

    @staticmethod
    async def test__when_filters_are_given__delete_every_match(
        stocked: MongoReadRepository[Machine], machine_writes: MongoWriteRepository[Machine]
    ) -> None:
        """The three lathes; the press and the mill stay."""
        deleted = await machine_writes.delete_by_filters([
            EqualityFilter(field="kind", value="lathe")
        ])

        remaining = await stocked.get_by_filters()

        assert deleted == 3
        assert [machine.id_ for machine in remaining] == ["m-3", "m-4"]

    @staticmethod
    async def test__when_no_filter_is_given__is_refused(
        stocked: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """It would delete everything; delete_all says so."""
        with pytest.raises(ValueError, match="delete_all"):
            await machine_writes.delete_by_filters([])

        assert await stocked.count_by_filters([]) == len(fleet)

    @staticmethod
    async def test__when_delete_all_is_called__all_are_deleted_and_counted(
        stocked: MongoReadRepository[Machine],
        machine_writes: MongoWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The collection is left empty."""
        deleted = await machine_writes.delete_all()

        assert deleted == len(fleet)
        assert await stocked.count_by_filters([]) == 0
