"""A double of Weaviate's async client and collection."""

import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import weaviate.classes as wvc
import weaviate.classes.query as wcq
from weaviate.collections.classes.aggregate import AggregateReturn
from weaviate.collections.classes.batch import (
    BatchObject,
    BatchObjectReturn,
    DeleteManyObject,
    DeleteManyReturn,
    ErrorObject,
)
from weaviate.collections.classes.internal import MetadataReturn, Object, QueryReturn


class WeaviateDouble:
    """One Weaviate collection held in memory, behind the async client's mocked calls."""

    def __init__(self) -> None:
        """Wire the mocked client and collection to the objects held."""
        self.objects: dict[str, Object[dict[str, Any], None]] = {}
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
    def stored(
        properties: dict[str, Any], vector: list[float] | None = None
    ) -> Object[dict[str, Any], None]:
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

    def by_id(self, id_: str) -> Object[dict[str, Any], None]:
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

    async def iterator(self) -> AsyncIterator[Object[dict[str, Any], None]]:
        """Iterate over every object held.

        Yields:
            Each object.
        """
        for item in list(self.objects.values()):
            yield item
