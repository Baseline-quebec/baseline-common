"""Fixtures for the MongoDB repositories: machines of a workshop, in a mongomock collection."""

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import mongomock
import pytest
from pydantic import BaseModel, Field
from pymongo import ReplaceOne
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.results import DeleteResult, InsertManyResult

from baseline_core.core.repository.nosql.mongo.mongo_read_repository import MongoReadRepository
from baseline_core.core.repository.nosql.mongo.mongo_write_repository import MongoWriteRepository
from tests.core.repository.conftest import Site


if TYPE_CHECKING:
    from pymongo.asynchronous.collection import AsyncCollection


class Machine(BaseModel):
    """A machine of a workshop, with a nested field, numbers, a datetime and a list."""

    id_: str | None = None
    name: str
    kind: str = "lathe"
    site: Site
    hours: float = 0.0
    shifts: int = 0
    serviced: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class MongomockCursor:
    """A mongomock cursor read with ``async for``, as a pymongo AsyncCursor is."""

    def __init__(self, cursor: Cursor[dict[str, Any]]) -> None:
        """Wrap the cursor.

        Args:
            cursor: The mongomock cursor.
        """
        self._cursor = cursor

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """Yield the documents.

        Yields:
            Each document.
        """
        for document in self._cursor:
            yield document


class AsyncMongomock:
    """A mongomock collection behind the async calls of a pymongo AsyncCollection."""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        """Wrap the collection.

        Args:
            collection: The mongomock collection.
        """
        self.name = collection.name
        self.collection = collection

    def find(self, query: dict[str, Any], **options: object) -> MongomockCursor:
        """Find documents.

        Args:
            query: The query.
            options: The sort, skip, limit or projection.

        Returns:
            Their cursor.
        """
        return MongomockCursor(self.collection.find(query, **options))

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        """Find a document.

        Args:
            query: The query.

        Returns:
            The document, or None.
        """
        return self.collection.find_one(query)

    async def count_documents(self, query: dict[str, Any]) -> int:
        """Count documents.

        Args:
            query: The query.

        Returns:
            The count.
        """
        return self.collection.count_documents(query)

    async def insert_many(self, documents: list[dict[str, Any]]) -> InsertManyResult:
        """Insert documents, in order.

        Args:
            documents: The documents.

        Returns:
            The result.
        """
        return self.collection.insert_many(documents)

    async def bulk_write(self, requests: Sequence[ReplaceOne[dict[str, Any]]]) -> None:
        """Replay a bulk of replacements one by one: mongomock 4.3 takes no pymongo 4.13+ one.

        Args:
            requests: The replacements.
        """
        for request in requests:
            self.collection.replace_one(
                request._filter, request._doc, upsert=bool(request._upsert)
            )

    async def delete_one(self, query: dict[str, Any]) -> DeleteResult:
        """Delete a document.

        Args:
            query: The query.

        Returns:
            The result.
        """
        return self.collection.delete_one(query)

    async def delete_many(self, query: dict[str, Any]) -> DeleteResult:
        """Delete documents.

        Args:
            query: The query.

        Returns:
            The result.
        """
        return self.collection.delete_many(query)


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
