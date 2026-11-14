"""Unit tests for the SQL writer: the write contract, each call in its savepoint."""

import uuid
from typing import cast

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from baseline_common.core.repository.abstract_repository import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_common.core.repository.query_filters import EqualityFilter
from baseline_common.core.repository.sql.abstract_rejection_translator import (
    RecordRejectedError,
    RejectionReason,
)
from baseline_common.core.repository.sql.sql_read_repository import SqlReadRepository
from baseline_common.core.repository.sql.sql_write_repository import SqlWriteRepository
from tests.core.repository.sql.workshop_tables import Machine, Part, Tool


class TestInsert:
    """An insert adds new rows and refuses an id already held."""

    @staticmethod
    async def test__many__are_inserted_in_order(
        machine_reads: SqlReadRepository[Machine],
        machine_writes: SqlWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Their ids, in order, and each read back."""
        inserted = await machine_writes.insert_many(fleet)

        assert inserted == [str(machine.id) for machine in fleet]
        assert len(await machine_reads.get_by_filters()) == len(fleet)

    @staticmethod
    async def test__an_id_held_or_given_twice__is_a_duplicate(
        machine_writes: SqlWriteRepository[Machine],
    ) -> None:
        """Refused before any write, as the contract says."""
        held = Machine(serial="S-1")
        await machine_writes.insert_one(held)
        twin = Machine(serial="S-2")

        with pytest.raises(DuplicateRecordError):
            await machine_writes.insert_one(Machine(id=held.id, serial="S-3"))

        with pytest.raises(DuplicateRecordError):
            await machine_writes.insert_many([twin, Machine(id=twin.id, serial="S-4")])

    @staticmethod
    async def test__a_primary_key_the_database_refuses__is_a_duplicate(
        session: AsyncSession, machine_writes: SqlWriteRepository[Machine]
    ) -> None:
        """Rows keyed by serial pass the check of held ids; the database refuses the taken id."""
        held = Machine(serial="S-1")
        await machine_writes.insert_one(held)
        session.expunge(held)
        by_serial = SqlWriteRepository(SqlReadRepository(session, Machine, id_field="serial"))

        with pytest.raises(DuplicateRecordError, match=str(held.id)):
            await by_serial.insert_one(Machine(id=held.id, serial="S-2"))


class TestRejection:
    """A row the database refuses raises ``RecordRejectedError``, and only its call is undone."""

    @staticmethod
    async def test__a_unique_value_taken__names_its_constraint_column_and_value(
        machine_writes: SqlWriteRepository[Machine],
    ) -> None:
        """The serial S-1 taken twice."""
        await machine_writes.insert_one(Machine(serial="S-1"))

        with pytest.raises(RecordRejectedError) as refused:
            await machine_writes.insert_one(Machine(serial="S-1"))

        assert refused.value.reason is RejectionReason.UNIQUE
        assert (refused.value.table, refused.value.constraint) == ("machine", "uq_machine_serial")
        assert (refused.value.columns, refused.value.values) == (["serial"], ["S-1"])
        assert refused.value.other_table is None

    @staticmethod
    async def test__a_unique_index__names_itself_its_column_and_value(
        session: AsyncSession,
    ) -> None:
        """The code T-1 taken twice, held unique by an index rather than a constraint."""
        tool_writes = SqlWriteRepository(SqlReadRepository(session, Tool))
        await tool_writes.insert_one(Tool(code="T-1"))

        with pytest.raises(RecordRejectedError) as refused:
            await tool_writes.insert_one(Tool(code="T-1"))

        assert refused.value.reason is RejectionReason.UNIQUE
        assert (refused.value.table, refused.value.constraint) == ("tool", "ix_tool_code")
        assert (refused.value.columns, refused.value.values) == (["code"], ["T-1"])

    @staticmethod
    async def test__a_check__names_its_constraint(
        machine_writes: SqlWriteRepository[Machine],
    ) -> None:
        """A negative size breaks the check."""
        with pytest.raises(RecordRejectedError) as refused:
            await machine_writes.insert_one(Machine(serial="S-1", size=-1))

        assert refused.value.reason is RejectionReason.CHECK
        assert refused.value.table == "machine"
        assert refused.value.constraint == "ck_machine_size_non_negative"
        assert (refused.value.columns, refused.value.other_table) == ([], None)

    @staticmethod
    async def test__a_missing_value__names_its_column(
        machine_writes: SqlWriteRepository[Machine],
    ) -> None:
        """A machine without a serial, which the column requires."""
        with pytest.raises(RecordRejectedError) as refused:
            await machine_writes.insert_one(Machine(serial=cast("str", None)))

        assert refused.value.reason is RejectionReason.REQUIRED
        assert (refused.value.table, refused.value.columns) == ("machine", ["serial"])

    @staticmethod
    async def test__a_value_no_constraint_names__is_invalid(
        machine_writes: SqlWriteRepository[Machine],
    ) -> None:
        """A serial longer than its column holds."""
        with pytest.raises(RecordRejectedError) as refused:
            await machine_writes.insert_one(Machine(serial="S" * 30))

        assert refused.value.reason is RejectionReason.INVALID_VALUE

    @staticmethod
    async def test__references__are_checked_both_ways(
        machine_writes: SqlWriteRepository[Machine], part_writes: SqlWriteRepository[Part]
    ) -> None:
        """A part of an unknown machine is refused; a machine with a part cannot be deleted."""
        machine = Machine(serial="S-1")
        await machine_writes.insert_one(machine)
        await part_writes.insert_one(Part(machine_id=machine.id))

        with pytest.raises(RecordRejectedError) as unknown:
            await part_writes.insert_one(Part(machine_id=uuid.uuid4()))

        with pytest.raises(RecordRejectedError) as used:
            await machine_writes.delete_one(str(machine.id))

        assert unknown.value.reason is RejectionReason.UNKNOWN_REFERENCE
        assert (unknown.value.table, unknown.value.other_table) == ("part", "machine")
        assert unknown.value.constraint == "fk_part_machine_id_machine"
        assert unknown.value.columns == ["machine_id"]
        assert used.value.reason is RejectionReason.STILL_REFERENCED
        assert (used.value.table, used.value.other_table) == ("part", "part")
        assert used.value.constraint == "fk_part_machine_id_machine"
        assert used.value.columns == ["id"]

    @staticmethod
    async def test__a_refused_call__leaves_the_earlier_ones(
        machine_reads: SqlReadRepository[Machine], machine_writes: SqlWriteRepository[Machine]
    ) -> None:
        """The second call is refused whole; the first one stays."""
        await machine_writes.insert_one(Machine(serial="S-1"))

        with pytest.raises(RecordRejectedError):
            await machine_writes.insert_many([Machine(serial="S-2"), Machine(serial="S-1")])

        assert [machine.serial for machine in await machine_reads.get_by_filters()] == ["S-1"]


class TestManyRows:
    """More ids than a statement takes: SQL Server allows 2100 parameters in one."""

    @staticmethod
    async def test__thousands_of_rows__are_inserted_updated_and_deleted(
        machine_reads: SqlReadRepository[Machine], machine_writes: SqlWriteRepository[Machine]
    ) -> None:
        """2200 machines, each call whole."""
        fleet = [Machine(serial=f"S-{number}") for number in range(2200)]

        ids = await machine_writes.insert_many(fleet)
        await machine_writes.update_many([
            Machine(id=machine.id, serial=machine.serial, name="mill") for machine in fleet
        ])
        renamed = await machine_reads.get_by_filters(
            [EqualityFilter(field="name", value="mill")], limit=None
        )
        deleted = await machine_writes.delete_many(ids)

        assert (len(renamed), deleted) == (2200, ids)
        assert await machine_reads.get_by_filters(limit=None) == []


class TestOtherErrors:
    """Only a refused row becomes ``RecordRejectedError``; any other database error stays itself."""

    @staticmethod
    async def test__a_missing_table__is_no_refusal(session: AsyncSession) -> None:
        """The tools' table dropped: the insert fails as SQLAlchemy raised it."""
        tool_writes = SqlWriteRepository(SqlReadRepository(session, Tool))
        await session.execute(text("DROP TABLE tool"))

        with pytest.raises(ProgrammingError):
            await tool_writes.insert_one(Tool(code="T-1"))


class TestUpdateAndUpsert:
    """An update replaces a held row; an upsert inserts or replaces."""

    @staticmethod
    async def test__an_upsert__replaces_and_keeps_what_the_database_filled(
        machine_reads: SqlReadRepository[Machine], machine_writes: SqlWriteRepository[Machine]
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
        machine_writes: SqlWriteRepository[Machine],
    ) -> None:
        """Refused before any write, as the contract says."""
        with pytest.raises(RecordNotFoundError):
            await machine_writes.update_one(Machine(serial="S-1"))

        with pytest.raises(RecordNotFoundError):
            await machine_writes.delete_one(str(uuid.uuid4()))
