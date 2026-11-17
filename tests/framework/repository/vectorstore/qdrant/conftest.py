"""Fixtures for the Qdrant repositories: workshops and their machines, in an in-memory Qdrant."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel
from qdrant_client import AsyncQdrantClient

from baseline_common.framework.repository.vectorstore.qdrant.qdrant_collection import (
    QdrantCollection,
)
from baseline_common.framework.repository.vectorstore.qdrant.qdrant_read_repository import (
    QdrantConfig,
    QdrantReadRepository,
)
from baseline_common.framework.repository.vectorstore.qdrant.qdrant_write_repository import (
    QdrantWriteRepository,
)
from tests.framework.repository.vectorstore.qdrant.machine import Machine


@pytest.fixture
async def qdrant_client() -> AsyncIterator[AsyncQdrantClient]:
    """Yield an in-memory Qdrant client.

    Yields:
        The client, closed afterwards.
    """
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


@pytest.fixture
def absent() -> AsyncMock:
    """Return a mocked client of a server that does not hold the collection.

    Returns:
        The client.
    """
    client = AsyncMock(spec=AsyncQdrantClient)
    client.collection_exists.return_value = False
    return client


@pytest.fixture
def present() -> AsyncMock:
    """Return a mocked client of a server that holds the collection.

    Returns:
        The client.
    """
    client = AsyncMock(spec=AsyncQdrantClient)
    client.collection_exists.return_value = True
    return client


@pytest.fixture
def embedder() -> Embedder:
    """Return an embedder of 8-dimension vectors that calls no provider.

    Returns:
        The embedder.
    """
    return Embedder(TestEmbeddingModel(dimensions=8))


@pytest.fixture
def config() -> QdrantConfig:
    """Return a collection config without sparse vectors, which need FastEmbed.

    Returns:
        The config.
    """
    return QdrantConfig(top_k=3, vector_size=8, use_sparse_vectors=False)


@pytest.fixture
def machine_reads(
    qdrant_client: AsyncQdrantClient, embedder: Embedder, config: QdrantConfig
) -> QdrantReadRepository[Machine]:
    """Return the reader of the machines.

    Returns:
        The reader.
    """
    return QdrantReadRepository(Machine, qdrant_client, embedder, config)


@pytest.fixture
def machine_writes(
    machine_reads: QdrantReadRepository[Machine],
    qdrant_client: AsyncQdrantClient,
    config: QdrantConfig,
) -> QdrantWriteRepository[Machine]:
    """Return the writer of the machines; the collection is created on the first write.

    Returns:
        The writer.
    """
    return QdrantWriteRepository(machine_reads, QdrantCollection(Machine, qdrant_client, config))


@pytest.fixture
def fleet() -> list[Machine]:
    """Return five machines of three workshops.

    Returns:
        The machines.
    """
    return [
        Machine(
            id_="m-1",
            name="Lathe A",
            workshop_id="w-1",
            tags=["north", "south"],
            hours=12.5,
            shifts=2,
            serviced=datetime(2026, 9, 1, tzinfo=UTC),
        ),
        Machine(
            id_="m-2",
            name="Lathe B",
            workshop_id="w-1",
            tags=["north"],
            hours=3.0,
            shifts=1,
            serviced=datetime(2026, 9, 20, tzinfo=UTC),
        ),
        Machine(
            id_="m-3",
            name="Press C",
            kind="press",
            workshop_id="w-2",
            tags=["south"],
            hours=12.5,
            shifts=3,
            serviced=datetime(2026, 9, 30, 10, tzinfo=UTC),
        ),
        Machine(id_="m-4", name="Mill D", kind="mill", workshop_id="w-2", hours=7.25, shifts=4),
        Machine(
            id_="m-5", name="Lathe E", workshop_id="w-3", serviced=datetime(2026, 9, 5, tzinfo=UTC)
        ),
    ]


@pytest.fixture
async def stocked(
    machine_reads: QdrantReadRepository[Machine],
    machine_writes: QdrantWriteRepository[Machine],
    fleet: list[Machine],
) -> QdrantReadRepository[Machine]:
    """Return the reader of the machines, the fleet inserted.

    Returns:
        The reader.
    """
    await machine_writes.insert_many(fleet)
    return machine_reads
