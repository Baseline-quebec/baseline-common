"""Fixtures for the Weaviate repositories: machines, in a Weaviate collection held in memory."""

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Annotated, Any, ClassVar, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
import weaviate.classes as wvc
import weaviate.classes.query as wcq
from pydantic import Field
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel
from weaviate.collections.classes.aggregate import AggregateReturn
from weaviate.collections.classes.batch import (
    BatchObject,
    BatchObjectReturn,
    DeleteManyObject,
    DeleteManyReturn,
    ErrorObject,
)
from weaviate.collections.classes.internal import MetadataReturn, Object, QueryReturn

from baseline_common.core.repository.vectorstore.vector_document import (
    VectorDocument,
    VectorStoreProperty,
)
from baseline_common.core.repository.vectorstore.weaviate.weaviate_read_repository import (
    TenantID,
    WeaviateConfig,
    WeaviateReadRepository,
)
from baseline_common.core.repository.vectorstore.weaviate.weaviate_write_repository import (
    WeaviateWriteRepository,
)


type WeaviateObject = Object[dict[str, Any], None]


class Machine(VectorDocument):
    """A machine of a workshop, with a field of each Weaviate data type."""

    __collection_name__: ClassVar[str] = "machines"
    __embedding_prefix__: ClassVar[str] = "machine: "

    name: Annotated[
        str,
        VectorStoreProperty(
            searchable=True,
            filterable=True,
            embedding_template="{value}",
            prompt_template="Machine: {value}",
        ),
    ]
    kind: Annotated[str, VectorStoreProperty(filterable=True)] = "lathe"
    mode: Literal["auto", "manual"] = "auto"
    tags: list[str] = Field(default_factory=list)
    hours: float | None = None
    shifts: int | None = None
    active: bool | None = None
    serviced: datetime | None = None


class WeaviateDouble:
    """One Weaviate collection held in memory, behind the async client's mocked calls."""

    def __init__(self) -> None:
        """Wire the mocked client and collection to the objects held."""
        self.objects: dict[str, WeaviateObject] = {}
        self.collection = MagicMock()
        self.collection.with_tenant.return_value = self.collection
        self.collection.query.fetch_objects = AsyncMock(side_effect=self.fetch_objects)
        self.collection.query.hybrid = AsyncMock(return_value=QueryReturn(objects=[]))
        self.collection.data.insert_many = AsyncMock(side_effect=self.insert_many)
        self.collection.data.update = AsyncMock(side_effect=self.update)
        self.collection.data.delete_many = AsyncMock(side_effect=self.delete_many)
        self.collection.aggregate.over_all = AsyncMock(side_effect=self.over_all)
        self.collection.iterator = self.iterator
        self.client = MagicMock()
        self.client.collections.get.return_value = self.collection
        self.client.collections.exists = AsyncMock(return_value=True)
        self.client.collections.create = AsyncMock()
        self.client.collections.delete = AsyncMock()

    @staticmethod
    def stored(properties: dict[str, Any], vector: list[float] | None = None) -> WeaviateObject:
        """Build an object as Weaviate returns it.

        Args:
            properties: Its properties; ``id_`` derives its UUID.
            vector: Its vector, under Weaviate's default name.

        Returns:
            The object.
        """
        return Object(
            uuid=uuid.uuid5(uuid.NAMESPACE_DNS, properties["id_"]),
            metadata=MetadataReturn(),
            properties=properties,
            references=None,
            vector={"default": vector} if vector else {},
            collection="Machines",
        )

    def by_id(self, id_: str) -> WeaviateObject:
        """Find a held object by its id_.

        Args:
            id_: The document's id_.

        Returns:
            The object.
        """
        return self.objects[str(uuid.uuid5(uuid.NAMESPACE_DNS, id_))]

    async def fetch_objects(
        self, *, filters: object = None, limit: int | None = None, **_options: object
    ) -> QueryReturn[dict[str, Any], None]:
        """Answer a fetch: an equality on the id_ is applied, any other filter returns all.

        Args:
            filters: The Weaviate filter.
            limit: The most objects returned.
            **_options: The offset, sort and vector options, recorded by the mock only.

        Returns:
            The objects.
        """
        found = list(self.objects.values())

        wanted = str(getattr(filters, "value", ""))

        if filters is not None and filters == wcq.Filter.by_property("id_").equal(wanted):
            found = [item for item in found if item.properties["id_"] == wanted]

        return QueryReturn(objects=found[:limit])

    async def insert_many(
        self, objects: Sequence[wvc.data.DataObject[Any, Any]]
    ) -> BatchObjectReturn:
        """Insert objects, refusing a UUID already held as Weaviate does.

        Args:
            objects: The objects.

        Returns:
            The UUIDs inserted and the errors, by position.
        """
        response = BatchObjectReturn()

        for index, data in enumerate(objects):
            if str(data.uuid) in self.objects:
                batch = BatchObject(collection="Machines", uuid=data.uuid, index=index)
                response.errors[index] = ErrorObject(message="id already exists", object_=batch)
                continue

            vector = list(data.vector) if isinstance(data.vector, list) else None
            self.objects[str(data.uuid)] = self.stored(data.properties, vector)
            response.uuids[index] = uuid.UUID(str(data.uuid))

        return response

    async def update(
        self, *, uuid: str, properties: dict[str, Any], vector: list[float] | None, **_: object
    ) -> None:
        """Replace an object's properties, and its vector when one is given.

        Args:
            uuid: The object's UUID.
            properties: Its new properties.
            vector: Its new vector; None keeps the stored one.
            **_: The references, unused.
        """
        held = self.objects[uuid]
        self.objects[uuid] = Object(
            uuid=held.uuid,
            metadata=held.metadata,
            properties=properties,
            references=None,
            vector=held.vector if vector is None else {"default": vector},
            collection=held.collection,
        )

    async def delete_many(self, *, where: object, verbose: bool) -> DeleteManyReturn[Any]:
        """Delete the objects of the UUIDs a ``contains_any`` filter lists.

        Args:
            where: The filter on the UUIDs.
            verbose: Whether each object deleted is listed.

        Returns:
            How many matched and were deleted.
        """
        uuids: list[object] = getattr(where, "value", [])
        held = [str(value) for value in uuids if str(value) in self.objects]

        for key in held:
            del self.objects[key]

        listed = [DeleteManyObject(uuid=uuid.UUID(key), successful=True) for key in held]
        return DeleteManyReturn(
            failed=0, matches=len(held), objects=listed if verbose else None, successful=len(held)
        )

    async def over_all(self, *, total_count: bool) -> AggregateReturn:
        """Count the objects held.

        Args:
            total_count: Whether the count is asked.

        Returns:
            The count.
        """
        return AggregateReturn(
            properties={}, total_count=len(self.objects) if total_count else None
        )

    async def iterator(self) -> AsyncIterator[WeaviateObject]:
        """Iterate over every object held.

        Yields:
            Each object.
        """
        for item in list(self.objects.values()):
            yield item


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
