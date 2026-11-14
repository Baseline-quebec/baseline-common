"""Unit tests for the SQL reader: the read contract, within a scope."""

from baseline_common.core.repository.query_filters import (
    ComparisonFilter,
    EqualityFilter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)
from baseline_common.core.repository.sql.sql_read_repository import SqlReadRepository
from baseline_common.core.repository.sql.sql_write_repository import SqlWriteRepository
from tests.core.repository.sql.workshop_tables import Machine


class TestGetById:
    """A row is found by its id, only within the reader's scope."""

    @staticmethod
    async def test__a_row__is_read_back_with_its_scope(
        machine_reads: SqlReadRepository[Machine], machine_writes: SqlWriteRepository[Machine]
    ) -> None:
        """The scope's site is set on insert, and the row comes back by its id."""
        (id_,) = await machine_writes.insert_many([Machine(serial="S-1")])

        found = await machine_reads.get_by_id(id_)

        assert found is not None
        assert (found.serial, found.site) == ("S-1", "Montréal")

    @staticmethod
    async def test__another_scope__sees_none_of_the_rows(
        machine_reads: SqlReadRepository[Machine], machine_writes: SqlWriteRepository[Machine]
    ) -> None:
        """A reader of another site finds neither the row nor any match."""
        (id_,) = await machine_writes.insert_many([Machine(serial="S-1")])
        other = SqlReadRepository(machine_reads.session, Machine, scope={"site": "Québec"})

        assert await other.get_by_id(id_) is None
        assert await other.get_by_filters() == []


class TestGetByFilters:
    """Filters compile to SQL; text compares regardless of case, as the contract says."""

    @staticmethod
    async def test__logical_and_comparison_filters__sort_and_page(
        machine_reads: SqlReadRepository[Machine],
        machine_writes: SqlWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The press or anything larger than 8, largest first, the second one only."""
        await machine_writes.insert_many(fleet)
        press_or_large = LogicalFilter(
            operator="|",
            operands=[
                EqualityFilter(field="name", value="PRESS"),
                ComparisonFilter(operator=">", field="size", value=8),
            ],
        )

        found = await machine_reads.get_by_filters(
            [press_or_large], sort_by="size", sort_order=SortOrder.DESC, limit=1, offset=1
        )

        assert [machine.serial for machine in found] == ["S-2"]

    @staticmethod
    async def test__a_text_filter__finds_a_part_of_the_text(
        machine_reads: SqlReadRepository[Machine],
        machine_writes: SqlWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """« RIL » is in « drill » only."""
        await machine_writes.insert_many(fleet)

        found = await machine_reads.get_by_filters([TextFilter(field="name", value="RIL")])

        assert [machine.serial for machine in found] == ["S-3"]

    @staticmethod
    async def test__pages_without_a_sort__are_in_a_stable_order(
        machine_reads: SqlReadRepository[Machine],
        machine_writes: SqlWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Three pages of one row hold the three rows, each once: they come in id order."""
        await machine_writes.insert_many(fleet)

        pages = [await machine_reads.get_by_filters(limit=1, offset=page) for page in range(3)]

        assert sorted(machine.serial for page in pages for machine in page) == [
            "S-1",
            "S-2",
            "S-3",
        ]
