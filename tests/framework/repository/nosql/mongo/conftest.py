"""Fixtures for the MongoDB repositories: machines of a workshop, in a mongomock collection."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import mongomock
import pytest
from pymongo.collection import Collection

from baseline_common.framework.repository.nosql.mongo.mongo_read_repository import (
    MongoReadRepository,
)
from baseline_common.framework.repository.nosql.mongo.mongo_write_repository import (
    MongoWriteRepository,
)
from tests.framework.repository.equipment import Site
from tests.framework.repository.nosql.mongo.async_mongomock import AsyncMongomock
from tests.framework.repository.nosql.mongo.machine import Machine


if TYPE_CHECKING:
    from pymongo.asynchronous.collection import AsyncCollection


@pytest.fixture
def stored() -> Collection[dict[str, Any]]:
    """Return an empty mongomock collection, to look at documents as stored.

    Returns:
        The collection, typed as the pymongo one it imitates.
    """
    return mongomock.MongoClient().workshop.machines


@pytest.fixture
def machine_reads(stored: Collection[dict[str, Any]]) -> MongoReadRepository[Machine]:
    """Return the reader of the machines.

    Returns:
        The reader.
    """
    collection = cast("AsyncCollection[dict[str, Any]]", AsyncMongomock(stored))
    return MongoReadRepository(Machine, collection, primary_key="id_")


@pytest.fixture
def machine_writes(machine_reads: MongoReadRepository[Machine]) -> MongoWriteRepository[Machine]:
    """Return the writer of the machines.

    Returns:
        The writer.
    """
    return MongoWriteRepository(machine_reads)


@pytest.fixture
def fleet() -> list[Machine]:
    """Return five machines of three sites.

    Returns:
        The machines.
    """
    return [
        Machine(
            id_="m-1",
            name="Lathe A",
            site=Site(city="Montréal"),
            tags=["north", "south"],
            hours=12.5,
            shifts=2,
            serviced=datetime(2026, 9, 1, tzinfo=UTC),
        ),
        Machine(
            id_="m-2",
            name="Lathe B",
            site=Site(city="Montréal"),
            tags=["north"],
            hours=3.0,
            shifts=1,
            serviced=datetime(2026, 9, 20, tzinfo=UTC),
        ),
        Machine(
            id_="m-3",
            name="Press C",
            kind="press",
            site=Site(city="Trois-Rivières"),
            tags=["south"],
            hours=12.5,
            shifts=3,
            serviced=datetime(2026, 9, 30, 10, tzinfo=UTC),
        ),
        Machine(
            id_="m-4", name="Mill D", kind="mill", site=Site(city="Laval"), hours=7.25, shifts=4
        ),
        Machine(
            id_="m-5",
            name="Lathe E",
            site=Site(city="Québec"),
            serviced=datetime(2026, 9, 5, tzinfo=UTC),
        ),
    ]


@pytest.fixture
async def stocked(
    machine_reads: MongoReadRepository[Machine],
    machine_writes: MongoWriteRepository[Machine],
    fleet: list[Machine],
) -> MongoReadRepository[Machine]:
    """Return the reader of the machines, the fleet inserted.

    Returns:
        The reader.
    """
    await machine_writes.insert_many(fleet)
    return machine_reads
