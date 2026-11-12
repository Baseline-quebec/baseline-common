"""WeaviateWriteRepository — writes documents to a Weaviate collection, embedding their text."""

import asyncio
import uuid
from collections.abc import Sequence
from itertools import starmap

import weaviate
import weaviate.classes as wvc
import weaviate.classes.query as wcq
from loguru import logger

from baseline_core.core.repository.abstract_repository import (
    AbstractWriteRepository,
    RecordNotFoundError,
)
from baseline_core.core.repository.vectorstore.vector_document import VectorDocument
from baseline_core.core.repository.vectorstore.weaviate.weaviate_read_repository import (
    WeaviateReadRepository,
)


class WeaviateWriteRepository[T: VectorDocument](AbstractWriteRepository[T]):
    """Writes one document type to its Weaviate collection, re-embedding only changed text."""

    def __init__(self, reads: WeaviateReadRepository[T]) -> None:
        """Bind the reader of the same collection.

        Args:
            reads: Reads what a write replaces; its collection, tenant and embedder are used.
        """
        self._reads = reads
        vector_name = reads.content_type.__vector_field_name__
        # The vector is sent beside the properties, never among them.
        self._unstored = {vector_name} if vector_name else set()

    async def insert_many(self, records: Sequence[T]) -> list[str]:
        """Embed and insert documents.

        Args:
            records: The documents.

        Returns:
            Their id_ values, in order.

        Raises:
            ValueError: When Weaviate refuses a document, an id_ already held among others.
        """
        data_objects = await asyncio.gather(*[
            self._data_object(document, include_vector=True) for document in records
        ])
        collection = self._reads.collection
        response = await collection.data.insert_many(data_objects)  # type: ignore[arg-type]

        if response.errors:
            refused = [
                f"{records[index].id_} ({error.message})"
                for index, error in sorted(response.errors.items())
            ]
            msg = f"Weaviate refused {len(refused)} of {len(records)}: {', '.join(refused)}"
            raise ValueError(msg)

        vectorstore_count = await collection.aggregate.over_all(total_count=True)
        logger.info(
            "Inserted {} documents into Weaviate, which holds {}",
            len(response.uuids),
            vectorstore_count.total_count,
        )
        return [document.id_ for document in records]

    async def update_many(self, records: Sequence[T]) -> list[str]:
        """Replace documents with the ones of the same id_.

        Args:
            records: The documents, as they now are.

        Returns:
            Their id_ values, in order.

        Raises:
            RecordNotFoundError: When an id_ is not held; nothing is written then.
        """
        held = await self._held(records)
        missing = [
            document.id_
            for document, existing in zip(records, held, strict=True)
            if existing is None
        ]

        if missing:
            msg = f"no document {', '.join(missing)}"
            raise RecordNotFoundError(msg)

        await asyncio.gather(*starmap(self._replace, zip(records, held, strict=True)))
        return [document.id_ for document in records]

    async def upsert_many(self, records: Sequence[T]) -> list[str]:
        """Insert documents, or replace those already in the collection.

        Args:
            records: The documents, as they now are; the last of an id_ repeated is kept.

        Returns:
            Their id_ values, in order.
        """
        latest = list({document.id_: document for document in records}.values())
        held = await self._held(latest)
        await asyncio.gather(*starmap(self._replace, zip(latest, held, strict=True)))
        return [document.id_ for document in records]

    async def delete_many(self, ids: list[str]) -> list[str]:
        """Delete documents.

        Args:
            ids: The documents' id_ values.

        Returns:
            The id_ values deleted.

        Raises:
            ValueError: When Weaviate refuses the delete or leaves a document undeleted.
        """
        uuids_to_delete = [self._uuid_of(id_) for id_ in ids]

        try:
            response = await self._reads.collection.data.delete_many(
                where=wcq.Filter.by_id().contains_any(uuids_to_delete), verbose=True
            )
        except (
            weaviate.exceptions.UnexpectedStatusCodeError,
            weaviate.exceptions.WeaviateQueryError,
        ) as err:
            raise ValueError(err.message) from err

        if len(uuids_to_delete) != response.successful:
            not_deleted_ids = list(
                set(uuids_to_delete) - {str(object_.uuid) for object_ in response.objects}
            )
            error_msg = (
                f"Failed to delete {len(not_deleted_ids)}/{len(uuids_to_delete)}. "
                f"These are the ids that were not deleted: {not_deleted_ids}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Deleted {} documents successfully from Weaviate.", response.successful)
        return ids

    async def delete_all(self) -> list[str]:
        """Delete every document of the collection.

        Returns:
            The id_ values deleted.
        """
        collection = self._reads.collection
        items = [item async for item in collection.iterator()]

        if not items:
            return []

        await collection.data.delete_many(
            where=wcq.Filter.by_id().contains_any([item.uuid for item in items]), verbose=True
        )
        return [item.properties.get("id_") for item in items]  # type: ignore[misc]

    async def _held(self, records: Sequence[T]) -> list[T | None]:
        """Read what the collection holds under each document's id_.

        Args:
            records: The documents.

        Returns:
            The document held under each id_, or None, in order.
        """
        return list(
            await asyncio.gather(*[self._reads.get_by_id(document.id_) for document in records])
        )

    async def _replace(self, document: T, existing: T | None) -> str:
        """Write a document, embedding it again only when it is new or its text to embed changed.

        Args:
            document: The document, as it now is.
            existing: The document the collection holds under its id_, or None to insert it.

        Returns:
            Its id_ value.
        """
        if existing is None:
            return await self.insert_one(document)

        data_object = await self._data_object(
            document, include_vector=existing.text_to_embed != document.text_to_embed
        )

        # No vector keeps the stored one: the text to embed did not change.
        await self._reads.collection.data.update(
            uuid=self._uuid_of(document.id_),
            properties=data_object.properties,
            references=None,
            vector=data_object.vector,
        )
        return document.id_

    async def _data_object(self, document: T, *, include_vector: bool) -> wvc.data.DataObject:
        """Turn a document into a Weaviate object, embedding its text when asked.

        Args:
            document: The document.
            include_vector: Whether the object carries a freshly computed vector.

        Returns:
            The Weaviate object, its UUID derived from the document's id_.
        """
        vector = None

        if include_vector:
            embedding_result = await self._reads.embedding_model.embed_query([
                document.text_to_embed
            ])
            vector = list(embedding_result.embeddings[0])

        properties = document.model_dump(exclude=self._unstored, exclude_none=True)
        return wvc.data.DataObject(
            properties=properties,  # type: ignore[arg-type]
            uuid=self._uuid_of(document.id_),
            vector=vector,
        )

    @staticmethod
    def _uuid_of(id_: str) -> str:
        """Derive the Weaviate UUID of a document, the same for its insert and its delete.

        Args:
            id_: The document's id_ value.

        Returns:
            The UUID, as text.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_))
