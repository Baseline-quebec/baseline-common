"""Unit tests for the MongoDB repository: one object that reads what it writes."""

from typing import TYPE_CHECKING, Any, cast

import pytest
from pymongo.collection import Collection

from baseline_common.framework.repository.abstract_repository import AbstractRepository
from baseline_common.framework.repository.nosql.mongo.mongo_repository import MongoRepository
from baseline_common.framework.repository.query_filters import EqualityFilter
from tests.framework.repository.nosql.mongo.async_mongomock import AsyncMongomock
from tests.framework.repository.nosql.mongo.machine import Machine


if TYPE_CHECKING:
    from pymongo.asynchronous.collection import AsyncCollection


@pytest.fixture
def machines(stored: Collection[dict[str, Any]]) -> MongoRepository[Machine]:
    """Return the repository of the machines.

    Returns:
        The repository.
    """
    collection = cast("AsyncCollection[dict[str, Any]]", AsyncMongomock(stored))
    return MongoRepository(Machine, collection, primary_key="id_")


class TestRepository:
    """The same object writes records and reads them back."""

    @staticmethod
    def test__when_the_repository_is_built__fulfills_both_contracts(
        machines: MongoRepository[Machine],
    ) -> None:
        """It is taken wherever a repository that reads and writes is."""
        assert isinstance(machines, AbstractRepository)

    @staticmethod
    async def test__when_records_are_written__reads_them_back(
        machines: MongoRepository[Machine], fleet: list[Machine]
    ) -> None:
        """Inserted, filtered, updated and deleted through one object."""
        await machines.insert_many(fleet)
        renamed = fleet[0].model_copy(update={"name": "Lathe Z"})

        await machines.update_one(renamed)
        await machines.delete_one("m-5")

        assert await machines.get_by_id("m-1") == renamed
        assert await machines.get_by_id("m-5") is None
        assert await machines.get_by_filters([EqualityFilter(field="kind", value="lathe")]) == [
            renamed,
            fleet[1],
        ]
