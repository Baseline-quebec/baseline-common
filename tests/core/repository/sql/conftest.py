"""Fixtures for the SQL repositories: machines and their parts, on each real database server."""

from collections.abc import AsyncIterator, Iterator

import pytest
from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from testcontainers.community.mssql import SqlServerContainer
from testcontainers.community.postgres import PostgresContainer

from baseline_common.core.repository.sql.sql_read_repository import SqlReadRepository
from baseline_common.core.repository.sql.sql_write_repository import SqlWriteRepository
from tests.core.repository.sql.workshop_tables import Machine, Part, WorkshopTable


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """Start PostgreSQL once; a container takes seconds to start.

    Yields:
        Its asyncpg URL.
    """
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture(scope="session")
def sqlserver_url() -> Iterator[str]:
    """Start SQL Server once, or skip where Microsoft's ODBC driver is not installed.

    Yields:
        Its aioodbc URL.
    """
    driver = "ODBC Driver 18 for SQL Server"
    pyodbc = pytest.importorskip("pyodbc", reason="unixODBC is missing", exc_type=ImportError)

    if driver not in pyodbc.drivers():
        pytest.skip(f"{driver} is missing; `uv run poe test-sqlserver` runs these tests")

    image = "mcr.microsoft.com/mssql/server:2022-latest"

    with SqlServerContainer(image, dialect="mssql+aioodbc") as sqlserver:
        url = make_url(sqlserver.get_connection_url())
        yield url.update_query_dict({
            "driver": driver,
            "TrustServerCertificate": "yes",
        }).render_as_string(hide_password=False)


@pytest.fixture(
    scope="session", params=["postgres_url", "sqlserver_url"], ids=["postgresql", "sqlserver"]
)
def database_url(request: pytest.FixtureRequest) -> str:
    """Each database server the repositories run on, in turn.

    Args:
        request: Names the server's fixture.

    Returns:
        The server's URL.
    """
    url: str = request.getfixturevalue(request.param)
    return url


@pytest.fixture
async def session(database_url: str) -> AsyncIterator[AsyncSession]:
    """A session on fresh tables, inside a transaction rolled back after the test.

    Args:
        database_url: The server.

    Yields:
        The session; its commits only release savepoints.
    """
    engine = create_async_engine(database_url, poolclass=NullPool)

    async with engine.connect() as connection:
        transaction = await connection.begin()
        await connection.run_sync(WorkshopTable.metadata.create_all)

        async with AsyncSession(
            bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
        ) as opened:
            yield opened

        await transaction.rollback()

    await engine.dispose()


@pytest.fixture
def machine_reads(session: AsyncSession) -> SqlReadRepository[Machine]:
    """Reads the machines of the Montréal site.

    Args:
        session: The test's session.

    Returns:
        The reader.
    """
    return SqlReadRepository(session, Machine, scope={"site": "Montréal"})


@pytest.fixture
def machine_writes(machine_reads: SqlReadRepository[Machine]) -> SqlWriteRepository[Machine]:
    """Writes the machines of the Montréal site.

    Args:
        machine_reads: The reader of the same scope.

    Returns:
        The writer.
    """
    return SqlWriteRepository(machine_reads)


@pytest.fixture
def part_writes(session: AsyncSession) -> SqlWriteRepository[Part]:
    """Writes the parts.

    Args:
        session: The test's session.

    Returns:
        The writer.
    """
    return SqlWriteRepository(SqlReadRepository(session, Part))


@pytest.fixture
def fleet() -> list[Machine]:
    """Three machines: a small lathe, a medium press, a large drill.

    Returns:
        The machines, smallest first.
    """
    return [
        Machine(serial="S-1", name="lathe", size=1),
        Machine(serial="S-2", name="Press", size=5),
        Machine(serial="S-3", name="drill", size=9),
    ]
