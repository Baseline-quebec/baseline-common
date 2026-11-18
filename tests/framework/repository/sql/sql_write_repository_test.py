"""Unit tests for the SQL writer: the write contract, each call in its savepoint."""

import uuid
from typing import cast

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from baseline_common.framework.repository.abstract_repository import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_common.framework.repository.query_filters import EqualityFilter
from baseline_common.framework.repository.sql.abstract_rejection_translator import (
    RecordRejectedError,
    RejectionReason,
)
from baseline_common.framework.repository.sql.sql_read_repository import SqlReadRepository
from baseline_common.framework.repository.sql.sql_write_repository import SqlWriteRepository
from tests.framework.repository.sql.workshop_tables import (
    Bench,
    Clamp,
    Hall,
    Machine,
    Part,
    Rack,
    Shift,
    Tool,
)


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

        with pytest.raises(DuplicateRecordError, match=f"already held: {held.id}"):
            await machine_writes.insert_one(Machine(id=held.id, serial="S-3"))

        with pytest.raises(DuplicateRecordError, match=f"already held: {twin.id}"):
            await machine_writes.insert_many([twin, Machine(id=twin.id, serial="S-4")])

    @staticmethod
    async def test__a_primary_key_the_database_refuses__is_a_duplicate(
        session: AsyncSession, machine_writes: SqlWriteRepository[Machine]
    ) -> None:
        """Rows keyed by serial pass the check of held ids; the database refuses the taken id."""
        held = Machine(serial="S-1")
        await machine_writes.insert_one(held)
        session.expunge(held)
        by_serial = SqlWriteRepository(SqlReadRepository(session, Machine, primary_key="serial"))

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


@pytest.fixture
async def quebec_machine(quebec_reads: SqlReadRepository[Machine]) -> Machine:
    """A lathe the Québec site holds.

    Args:
        quebec_reads: The reader of the Québec site.

    Returns:
        The machine, as inserted.
    """
    machine = Machine(serial="S-9", name="lathe")
    await SqlWriteRepository(quebec_reads).insert_one(machine)
    return machine


class TestScope:
    """A scoped write changes the rows of its scope and never moves or deletes another one's."""

    @staticmethod
    async def test__an_id_held_in_another_scope__is_refused_and_its_row_is_kept(
        machine_reads: SqlReadRepository[Machine],
        machine_writes: SqlWriteRepository[Machine],
        quebec_reads: SqlReadRepository[Machine],
        quebec_machine: Machine,
    ) -> None:
        """Montréal's upsert of Québec's id is a duplicate, its update a row not found."""
        held = f"already held: {quebec_machine.id}"

        with pytest.raises(DuplicateRecordError, match=held):
            await machine_writes.upsert_one(
                Machine(id=quebec_machine.id, serial="S-9", name="mill")
            )

        with pytest.raises(RecordNotFoundError, match=f"no record {quebec_machine.id}"):
            await machine_writes.update_one(
                Machine(id=quebec_machine.id, serial="S-9", name="mill")
            )

        with pytest.raises(DuplicateRecordError, match=held):
            await machine_writes.upsert_many([
                Machine(serial="S-1"),
                Machine(id=quebec_machine.id, serial="S-9", name="mill"),
            ])

        stored = await quebec_reads.get_by_id(str(quebec_machine.id))
        assert stored is not None
        assert (stored.site, stored.name) == ("Québec", "lathe")
        assert await machine_reads.get_by_filters() == []

    @staticmethod
    async def test__an_id_held_in_the_scope__is_replaced(
        machine_reads: SqlReadRepository[Machine],
        machine_writes: SqlWriteRepository[Machine],
        quebec_reads: SqlReadRepository[Machine],
        quebec_machine: Machine,
    ) -> None:
        """Montréal updates and upserts its own row, beside Québec's."""
        machine = Machine(serial="S-1")
        await machine_writes.insert_one(machine)

        await machine_writes.update_one(Machine(id=machine.id, serial="S-1", name="mill"))
        await machine_writes.upsert_many([
            Machine(id=machine.id, serial="S-1", name="drill"),
            Machine(serial="S-2"),
        ])

        stored = await machine_reads.get_by_id(str(machine.id))
        assert stored is not None
        assert (stored.site, stored.name) == ("Montréal", "drill")
        assert len(await machine_reads.get_by_filters()) == 2
        kept = await quebec_reads.get_by_id(str(quebec_machine.id))
        assert kept is not None
        assert (kept.site, kept.name) == ("Québec", "lathe")

    @staticmethod
    async def test__a_repository_id_other_than_the_primary_key__moves_no_row(
        session: AsyncSession, quebec_reads: SqlReadRepository[Machine], quebec_machine: Machine
    ) -> None:
        """Merge matches the primary key, not the serial the repository names rows by."""
        by_serial = SqlWriteRepository(
            SqlReadRepository(session, Machine, primary_key="serial", scope={"site": "Montréal"})
        )

        with pytest.raises(DuplicateRecordError, match=f"already held: {quebec_machine.id}"):
            await by_serial.upsert_one(Machine(id=quebec_machine.id, serial="S-NEW"))

        kept = await quebec_reads.get_by_id(str(quebec_machine.id))
        assert kept is not None
        assert (kept.site, kept.serial) == ("Québec", "S-9")

    @staticmethod
    async def test__a_primary_key_holding_the_scope__is_the_scope_s_own(
        session: AsyncSession,
    ) -> None:
        """The scope's serial takes the given one's place in the key, so nothing moves."""
        first = SqlReadRepository(session, Shift, "rank", scope={"machine_serial": "S-1"})
        second = SqlReadRepository(session, Shift, "rank", scope={"machine_serial": "S-2"})
        await SqlWriteRepository(first).insert_one(Shift(machine_serial="", rank=1))

        await SqlWriteRepository(second).upsert_one(Shift(machine_serial="S-1", rank=1))

        assert [shift.rank for shift in await first.get_by_filters()] == [1]
        assert [shift.rank for shift in await second.get_by_filters()] == [1]

    @staticmethod
    async def test__an_owned_child_held_in_another_scope__is_refused_and_kept(
        session: AsyncSession,
    ) -> None:
        """A Montréal hall cannot take a Québec hall's bench; it still rewrites its own."""
        quebec = SqlWriteRepository(SqlReadRepository(session, Hall, scope={"site": "Québec"}))
        montreal = SqlWriteRepository(SqlReadRepository(session, Hall, scope={"site": "Montréal"}))
        vise, anvil = Bench(name="vise"), Bench(name="anvil")
        quebec_hall, montreal_hall = Hall(benches=[vise]), Hall(benches=[anvil])
        await quebec.insert_one(quebec_hall)
        await montreal.insert_one(montreal_hall)
        held = f"already held: {vise.id}"

        with pytest.raises(DuplicateRecordError, match=held):
            await montreal.upsert_one(Hall(benches=[Bench(id=vise.id, name="moved")]))

        with pytest.raises(DuplicateRecordError, match=held):
            await montreal.update_one(
                Hall(id=montreal_hall.id, benches=[Bench(id=vise.id, name="moved")])
            )

        await montreal.upsert_one(
            Hall(
                id=montreal_hall.id,
                benches=[Bench(id=anvil.id, hall_id=montreal_hall.id, name="press")],
            )
        )

        benches = SqlReadRepository(session, Bench)
        kept, renamed = (
            await benches.get_by_id(str(vise.id)),
            await benches.get_by_id(str(anvil.id)),
        )
        assert kept is not None
        assert renamed is not None
        assert (kept.hall_id, kept.name) == (quebec_hall.id, "vise")
        assert (renamed.hall_id, renamed.name) == (montreal_hall.id, "press")

    @staticmethod
    async def test__a_grandchild_held_in_another_scope__is_refused_and_kept(
        session: AsyncSession,
    ) -> None:
        """A clamp is in the scope of its bench's hall; Montréal still rewrites its own clamp."""
        quebec = SqlWriteRepository(SqlReadRepository(session, Hall, scope={"site": "Québec"}))
        montreal = SqlWriteRepository(SqlReadRepository(session, Hall, scope={"site": "Montréal"}))
        quebec_clamp, montreal_clamp = Clamp(name="vise"), Clamp(name="vise")
        quebec_bench = Bench(clamps=[quebec_clamp])
        await quebec.insert_one(Hall(benches=[quebec_bench]))
        bench = Bench(clamps=[montreal_clamp])
        hall = Hall(benches=[bench])
        await montreal.insert_one(hall)

        with pytest.raises(DuplicateRecordError, match=f"already held: {quebec_clamp.id}"):
            await montreal.upsert_one(Hall(benches=[Bench(clamps=[Clamp(id=quebec_clamp.id)])]))

        await montreal.upsert_one(
            Hall(
                id=hall.id,
                benches=[
                    Bench(
                        id=bench.id,
                        hall_id=hall.id,
                        clamps=[Clamp(id=montreal_clamp.id, bench_id=bench.id, name="press")],
                    )
                ],
            )
        )

        clamps = SqlReadRepository(session, Clamp)
        kept = await clamps.get_by_id(str(quebec_clamp.id))
        renamed = await clamps.get_by_id(str(montreal_clamp.id))
        assert kept is not None
        assert renamed is not None
        assert (kept.bench_id, kept.name) == (quebec_bench.id, "vise")
        assert (renamed.bench_id, renamed.name) == (bench.id, "press")

    @staticmethod
    async def test__an_owned_row_of_the_same_table__is_held_in_its_own_scope(
        session: AsyncSession,
    ) -> None:
        """A Montréal rack cannot take the rack stacked on a Québec one."""
        racks = SqlReadRepository(session, Rack, scope={"site": "Québec"})
        top = Rack(site="Québec")
        base = Rack(stacked=[top])
        await SqlWriteRepository(racks).insert_one(base)
        montreal = SqlWriteRepository(SqlReadRepository(session, Rack, scope={"site": "Montréal"}))

        with pytest.raises(DuplicateRecordError, match=f"already held: {top.id}"):
            await montreal.upsert_one(Rack(stacked=[Rack(id=top.id, site="Montréal")]))

        kept = await racks.get_by_id(str(top.id))
        assert kept is not None
        assert kept.base_id == base.id

    @staticmethod
    async def test__an_id_held_in_another_scope__is_not_found_and_nothing_is_deleted(
        machine_writes: SqlWriteRepository[Machine],
        quebec_reads: SqlReadRepository[Machine],
        quebec_machine: Machine,
    ) -> None:
        """Through Montréal, Québec's id is refused as an id held nowhere, before any delete."""
        machine = Machine(serial="S-1")
        await machine_writes.insert_one(machine)

        with pytest.raises(RecordNotFoundError, match=str(quebec_machine.id)):
            await machine_writes.delete_one(str(quebec_machine.id))

        with pytest.raises(RecordNotFoundError, match=str(quebec_machine.id)):
            await machine_writes.delete_many([str(machine.id), str(quebec_machine.id)])

        assert await quebec_reads.get_by_id(str(quebec_machine.id)) is not None
        assert await machine_writes.delete_one(str(machine.id)) == str(machine.id)

    @staticmethod
    async def test__an_unscoped_writer__deletes_a_row_of_any_scope(
        session: AsyncSession, quebec_reads: SqlReadRepository[Machine], quebec_machine: Machine
    ) -> None:
        """Without a scope, a delete removes the row of the id, as it always did."""
        writes = SqlWriteRepository(SqlReadRepository(session, Machine))

        deleted = await writes.delete_one(str(quebec_machine.id))

        assert deleted == str(quebec_machine.id)
        assert await quebec_reads.get_by_id(str(quebec_machine.id)) is None

    @staticmethod
    async def test__an_unscoped_writer__replaces_a_row_of_any_scope(
        session: AsyncSession, quebec_machine: Machine
    ) -> None:
        """Without a scope, an upsert or an update replaces the row of the id, as it always did."""
        reads = SqlReadRepository(session, Machine)
        writes = SqlWriteRepository(reads)

        await writes.upsert_one(Machine(id=quebec_machine.id, serial="S-9", name="mill"))
        await writes.update_one(
            Machine(id=quebec_machine.id, serial="S-9", name="drill", site="Lévis")
        )

        stored = await reads.get_by_id(str(quebec_machine.id))
        assert stored is not None
        assert (stored.site, stored.name) == ("Lévis", "drill")
