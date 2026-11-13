"""Unit tests for the PostgreSQL writer: the write contract, each call in its savepoint."""

import uuid

import pytest

from baseline_common.core.repository.abstract_repository import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_common.core.repository.sql.postgres.postgres_read_repository import (
    PostgresReadRepository,
)
from baseline_common.core.repository.sql.postgres.postgres_write_repository import (
    PostgresWriteRepository,
    RecordRejectedError,
    RejectionReason,
)
from tests.core.repository.sql.postgres.conftest import Machine, Part


class TestInsert:
    """An insert adds new rows and refuses an id already held."""

    @staticmethod
    async def test__many__are_inserted_in_order(
        machine_reads: PostgresReadRepository[Machine],
        machine_writes: PostgresWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Their ids, in order, and each read back."""
        inserted = await machine_writes.insert_many(fleet)

        assert inserted == [str(machine.id) for machine in fleet]
        assert len(await machine_reads.get_by_filters()) == len(fleet)

    @staticmethod
    async def test__an_id_held_or_given_twice__is_a_duplicate(
        machine_writes: PostgresWriteRepository[Machine],
    ) -> None:
        """Refused before any write, as the contract says."""
        held = Machine(serial="S-1")
        await machine_writes.insert_one(held)
        twin = Machine(serial="S-2")

        with pytest.raises(DuplicateRecordError):
            await machine_writes.insert_one(Machine(id=held.id, serial="S-3"))

        with pytest.raises(DuplicateRecordError):
            await machine_writes.insert_many([twin, Machine(id=twin.id, serial="S-4")])


class TestRejection:
    """A row the database refuses raises ``RecordRejectedError``, and only its call is undone."""

    @staticmethod
    async def test__a_unique_value_taken__names_its_constraint_column_and_value(
        machine_writes: PostgresWriteRepository[Machine],
    ) -> None:
        """The serial S-1 taken twice."""
        await machine_writes.insert_one(Machine(serial="S-1"))

        with pytest.raises(RecordRejectedError) as refused:
            await machine_writes.insert_one(Machine(serial="S-1"))

        assert refused.value.reason is RejectionReason.UNIQUE
        assert refused.value.constraint == "uq_machine_serial"
        assert (refused.value.columns, refused.value.values) == (["serial"], ["S-1"])

    @staticmethod
    async def test__a_check__names_its_constraint(
        machine_writes: PostgresWriteRepository[Machine],
    ) -> None:
        """A negative size breaks the check."""
        with pytest.raises(RecordRejectedError) as refused:
            await machine_writes.insert_one(Machine(serial="S-1", size=-1))

        assert refused.value.reason is RejectionReason.CHECK
        assert refused.value.constraint == "ck_machine_size_non_negative"

    @staticmethod
    async def test__references__are_checked_both_ways(
        machine_writes: PostgresWriteRepository[Machine],
        part_writes: PostgresWriteRepository[Part],
    ) -> None:
        """A part of an unknown machine is refused; a machine with a part cannot be deleted."""
        machine = Machine(serial="S-1")
        await machine_writes.insert_one(machine)
        await part_writes.insert_one(Part(machine_id=machine.id))

        with pytest.raises(RecordRejectedError) as unknown:
            await part_writes.insert_one(Part(machine_id=uuid.uuid4()))

        with pytest.raises(RecordRejectedError) as used:
            await machine_writes.delete_one(str(machine.id))

        assert (unknown.value.reason, unknown.value.columns) == (
            RejectionReason.UNKNOWN_REFERENCE,
            ["machine_id"],
        )
        assert (used.value.reason, used.value.other_table) == (
            RejectionReason.STILL_REFERENCED,
            "part",
        )

    @staticmethod
    async def test__a_refused_call__leaves_the_earlier_ones(
        machine_reads: PostgresReadRepository[Machine],
        machine_writes: PostgresWriteRepository[Machine],
    ) -> None:
        """The second call is refused whole; the first one stays."""
        await machine_writes.insert_one(Machine(serial="S-1"))

        with pytest.raises(RecordRejectedError):
            await machine_writes.insert_many([Machine(serial="S-2"), Machine(serial="S-1")])

        assert [machine.serial for machine in await machine_reads.get_by_filters()] == ["S-1"]


class TestUpdateAndUpsert:
    """An update replaces a held row; an upsert inserts or replaces."""

    @staticmethod
    async def test__an_upsert__replaces_and_keeps_what_the_database_filled(
        machine_reads: PostgresReadRepository[Machine],
        machine_writes: PostgresWriteRepository[Machine],
    ) -> None:
        """The new name is kept; the creation time the server set survives the replacement."""
        machine = Machine(serial="S-1")
        await machine_writes.insert_one(machine)
        created = (await machine_reads.get_by_id(str(machine.id)) or machine).created_at

        await machine_writes.upsert_one(Machine(id=machine.id, serial="S-1", name="mill"))

        stored = await machine_reads.get_by_id(str(machine.id))
        assert stored is not None
        assert (stored.name, stored.created_at) == ("mill", created)

    @staticmethod
    async def test__a_row_not_held__cannot_be_updated_or_deleted(
        machine_writes: PostgresWriteRepository[Machine],
    ) -> None:
        """Refused before any write, as the contract says."""
        with pytest.raises(RecordNotFoundError):
            await machine_writes.update_one(Machine(serial="S-1"))

        with pytest.raises(RecordNotFoundError):
            await machine_writes.delete_one(str(uuid.uuid4()))
