"""Fixtures for the Weaviate repositories: machines, in a Weaviate collection held in memory."""

import pytest
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

from baseline_common.framework.repository.vectorstore.weaviate.weaviate_read_repository import (
    TenantID,
    WeaviateConfig,
    WeaviateReadRepository,
)
from baseline_common.framework.repository.vectorstore.weaviate.weaviate_write_repository import (
    WeaviateWriteRepository,
)
from tests.framework.repository.vectorstore.weaviate.machine import Machine
from tests.framework.repository.vectorstore.weaviate.weaviate_double import WeaviateDouble


@pytest.fixture
def weaviate_double() -> WeaviateDouble:
    """Return an empty collection held in memory.

    Returns:
        The double.
    """
    return WeaviateDouble()


@pytest.fixture
def embedder() -> Embedder:
    """Return an embedder of 4-dimension vectors that calls no provider.

    Returns:
        The embedder.
    """
    return Embedder(TestEmbeddingModel(dimensions=4))


@pytest.fixture
def config() -> WeaviateConfig:
    """Return a search config for a collection without tenants.

    Returns:
        The config.
    """
    return WeaviateConfig(top_k=3, hybrid_weight=0.7, tenant=TenantID(""))


@pytest.fixture
def machine_reads(
    weaviate_double: WeaviateDouble, embedder: Embedder, config: WeaviateConfig
) -> WeaviateReadRepository[Machine]:
    """Return the reader of the machines.

    Returns:
        The reader.
    """
    return WeaviateReadRepository(Machine, weaviate_double.client, embedder, config)


@pytest.fixture
def machine_writes(
    machine_reads: WeaviateReadRepository[Machine],
) -> WeaviateWriteRepository[Machine]:
    """Return the writer of the machines.

    Returns:
        The writer.
    """
    return WeaviateWriteRepository(machine_reads)


@pytest.fixture
def fleet() -> list[Machine]:
    """Return three machines.

    Returns:
        The machines.
    """
    return [
        Machine(id_="m-1", name="Lathe A", hours=12.5, shifts=2, active=True),
        Machine(id_="m-2", name="Press B", kind="press", tags=["north"]),
        Machine(id_="m-3", name="Mill C", kind="mill", mode="manual"),
    ]
