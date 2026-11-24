"""Unit tests for the Weaviate repository: one object that reads what it writes."""

import pytest
from pydantic_ai import Embedder

from baseline_common.framework.repository.abstract_repository import AbstractRepository
from baseline_common.framework.repository.vectorstore.weaviate.weaviate_read_repository import (
    WeaviateConfig,
)
from baseline_common.framework.repository.vectorstore.weaviate.weaviate_repository import (
    WeaviateRepository,
)
from tests.framework.repository.vectorstore.weaviate.machine import Machine
from tests.framework.repository.vectorstore.weaviate.weaviate_double import WeaviateDouble


@pytest.fixture
def machines(
    weaviate_double: WeaviateDouble, embedder: Embedder, config: WeaviateConfig
) -> WeaviateRepository[Machine]:
    """Return the repository of the machines.

    Returns:
        The repository.
    """
    return WeaviateRepository(Machine, weaviate_double.client, embedder, config)


class TestRepository:
    """The same object writes documents and reads them back."""

    @staticmethod
    def test__when_the_repository_is_built__returns_an_abstract_repository(
        machines: WeaviateRepository[Machine],
    ) -> None:
        """It is taken wherever a repository that reads and writes is."""
        assert isinstance(machines, AbstractRepository)

    @staticmethod
    async def test__when_documents_are_written__returns_what_was_written(
        machines: WeaviateRepository[Machine], fleet: list[Machine]
    ) -> None:
        """Inserted, updated and deleted through one object."""
        await machines.insert_many(fleet)
        renamed = fleet[0].model_copy(update={"name": "Lathe Z"})

        await machines.update_one(renamed)
        await machines.delete_one("m-2")

        assert await machines.get_by_id("m-1") == renamed
        assert await machines.get_by_id("m-2") is None
        assert await machines.get_by_id("m-3") == fleet[2]
