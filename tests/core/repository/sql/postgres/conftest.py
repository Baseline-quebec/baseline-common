"""Fixtures for the PostgreSQL repositories: machines and their parts, on a real server."""

import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import ClassVar

import pytest
from sqlalchemy import CheckConstraint, DateTime, MetaData, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import Field, SQLModel
from testcontainers.community.postgres import PostgresContainer

from baseline_common.core.repository.sql.postgres.postgres_read_repository import (
    PostgresReadRepository,
)
from baseline_common.core.repository.sql.postgres.postgres_write_repository import (
    PostgresWriteRepository,
)


class WorkshopTable(SQLModel):
    """Base of the test tables: a metadata of their own, created inside each test's transaction."""

    metadata: ClassVar[MetaData] = MetaData(
        naming_convention={
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Machine(WorkshopTable, table=True):
    """A machine of a site: a unique serial, a size that cannot be negative, a server timestamp."""

    __table_args__ = (CheckConstraint("size >= 0", name="size_non_negative"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    serial: str = Field(unique=True)
    name: str = "lathe"
    size: int = 1
    site: str = ""
    created_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={"server_default": func.now(), "nullable": False},
    )


class Part(WorkshopTable, table=True):
    """A part of a machine; the machine cannot be deleted while it has parts."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    machine_id: uuid.UUID = Field(foreign_key="machine.id", ondelete="RESTRICT")


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """Start PostgreSQL once; a container takes seconds to start.

    Yields:
        Its asyncpg URL.
    """
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
async def session(postgres_url: str) -> AsyncIterator[AsyncSession]:
    """A session on fresh tables, inside a transaction rolled back after the test.

    Args:
        postgres_url: The server.

    Yields:
        The session; its commits only release savepoints.
    """
    engine = create_async_engine(postgres_url, poolclass=NullPool)

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
def machine_reads(session: AsyncSession) -> PostgresReadRepository[Machine]:
    """Reads the machines of the Montréal site.

    Args:
        session: The test's session.

    Returns:
        The reader.
    """
    return PostgresReadRepository(session, Machine, scope={"site": "Montréal"})


@pytest.fixture
def machine_writes(
    machine_reads: PostgresReadRepository[Machine],
) -> PostgresWriteRepository[Machine]:
    """Writes the machines of the Montréal site.

    Args:
        machine_reads: The reader of the same scope.

    Returns:
        The writer.
    """
    return PostgresWriteRepository(machine_reads)


@pytest.fixture
def part_writes(session: AsyncSession) -> PostgresWriteRepository[Part]:
    """Writes the parts.

    Args:
        session: The test's session.

    Returns:
        The writer.
    """
    return PostgresWriteRepository(PostgresReadRepository(session, Part))


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
