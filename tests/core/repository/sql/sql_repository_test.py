"""Unit tests for the SQL repository: one object that reads what it writes."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from baseline_common.core.repository.abstract_repository import AbstractRepository
from baseline_common.core.repository.sql.sql_repository import SqlRepository
from tests.core.repository.sql.workshop_tables import Machine


@pytest.fixture
def machines(session: AsyncSession) -> SqlRepository[Machine]:
    """Reads and writes the machines of the Montréal site.

    Args:
        session: The test's session.

    Returns:
        The repository.
    """
    return SqlRepository(session, Machine, scope={"site": "Montréal"})


class TestRepository:
    """The same object writes rows and reads them back, within its scope."""

    @staticmethod
    def test__the_repository__fulfills_both_contracts(machines: SqlRepository[Machine]) -> None:
        """It is taken wherever a repository that reads and writes is."""
        assert isinstance(machines, AbstractRepository)

    @staticmethod
    async def test__what_it_writes__it_reads_back(
        machines: SqlRepository[Machine], fleet: list[Machine]
    ) -> None:
        """Inserted in the scope, updated and deleted through one object."""
        ids = await machines.insert_many(fleet)

        await machines.update_one(Machine(id=fleet[0].id, serial="S-1", name="mill"))
        await machines.delete_one(ids[1])

        renamed = await machines.get_by_id(ids[0])
        held = await machines.get_by_filters()
        assert renamed is not None
        assert (renamed.name, renamed.site) == ("mill", "Montréal")
        assert await machines.get_by_id(ids[1]) is None
        assert sorted(machine.serial for machine in held) == ["S-1", "S-3"]
