"""Unit tests for the Qdrant repository: one object that reads what it writes."""

import pytest
from pydantic_ai import Embedder
from qdrant_client import AsyncQdrantClient

from baseline_common.core.repository.abstract_repository import AbstractRepository
from baseline_common.core.repository.query_filters import EqualityFilter
from baseline_common.core.repository.vectorstore.qdrant.qdrant_read_repository import QdrantConfig
from baseline_common.core.repository.vectorstore.qdrant.qdrant_repository import QdrantRepository
from tests.core.repository.vectorstore.qdrant.conftest import Machine


@pytest.fixture
def machines(
    qdrant_client: AsyncQdrantClient, embedder: Embedder, config: QdrantConfig
) -> QdrantRepository[Machine]:
    """Return the repository of the machines; the collection is created on the first write.

    Returns:
        The repository.
    """
    return QdrantRepository(Machine, qdrant_client, embedder, config)


class TestRepository:
    """The same object writes documents and reads them back."""

    @staticmethod
    def test__the_repository__fulfills_both_contracts(machines: QdrantRepository[Machine]) -> None:
        """It is taken wherever a repository that reads and writes is."""
        assert isinstance(machines, AbstractRepository)

    @staticmethod
    async def test__what_it_writes__it_reads_back(
        machines: QdrantRepository[Machine], fleet: list[Machine]
    ) -> None:
        """Inserted, filtered, updated and deleted through one object."""
        await machines.insert_many(fleet)
        renamed = fleet[0].model_copy(update={"name": "Lathe Z"})

        await machines.update_one(renamed)
        await machines.delete_one("m-2")

        assert await machines.get_by_id("m-1") == renamed
        assert await machines.get_by_id("m-2") is None
        assert await machines.get_by_filters(
            [EqualityFilter(field="workshop_id", value="w-2")], sort_by="hours"
        ) == [fleet[3], fleet[2]]
