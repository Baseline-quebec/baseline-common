"""QdrantWriteRepository — writes documents to a Qdrant collection, embedding their text."""

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import Any

from loguru import logger
from qdrant_client.models import Document, FilterSelector, PointIdsList, PointStruct
from qdrant_client.models import Filter as QdrantFilter

from baseline_core.core.repository.abstract_repository import (
    AbstractWriteRepository,
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_core.core.repository.query_filters import Filter
from baseline_core.core.repository.vectorstore.qdrant.qdrant_collection import QdrantCollection
from baseline_core.core.repository.vectorstore.qdrant.qdrant_read_repository import (
    QdrantReadRepository,
)
from baseline_core.core.repository.vectorstore.vector_document import VectorDocument


class QdrantWriteRepository[T: VectorDocument](AbstractWriteRepository[T]):
    """Writes one document type to its Qdrant collection, re-embedding only changed text."""

    def __init__(self, reads: QdrantReadRepository[T], collection: QdrantCollection) -> None:
        """Bind the reader and the collection of the same document type.

        Args:
            reads: Reads what a write replaces; its client, embedder and config are used.
            collection: Creates the collection on the first write.
        """
        self._reads = reads
        self._collection = collection
        vector_name = reads.content_type.__vector_field_name__
        self._unstored = {vector_name} if vector_name else set()

    async def insert_many(self, records: Sequence[T]) -> list[str]:
        """Embed and insert documents, creating the collection if need be.

        Args:
            records: The documents.

        Returns:
            Their id_ values, in order.

        Raises:
            DuplicateRecordError: When an id_ is already held, or given twice.
        """
        ids = [document.id_ for document in records]

        if not ids:
            return []

        await self._collection.create_collection()
        held = await self._held(ids)
        repeated = {id_ for id_, times in Counter(ids).items() if times > 1} | held.keys()

        if repeated:
            msg = f"already held: {', '.join(sorted(repeated))}"
            raise DuplicateRecordError(msg)

        await self._upsert_points(records)
        total = await self._reads.count_by_filters([])
        logger.info(
            "Inserted {} documents into {}, which holds {}",
            len(ids),
            self._reads.collection_name,
            total,
        )
        return ids

    async def update_many(self, records: Sequence[T]) -> list[str]:
        """Replace documents with the ones of the same id_.

        Args:
            records: The documents, as they now are.

        Returns:
            Their id_ values, in order.

        Raises:
            RecordNotFoundError: When the collection holds no document of an id_.
        """
        ids = [document.id_ for document in records]
        held = await self._held(ids)
        missing = [id_ for id_ in ids if id_ not in held]

        if missing:
            msg = f"no document {', '.join(missing)}"
            raise RecordNotFoundError(msg)

        await self._replace(records, held)
        return ids

    async def upsert_many(self, records: Sequence[T]) -> list[str]:
        """Insert documents, or replace those already in the collection.

        Args:
            records: The documents, as they now are; the last of an id_ repeated is kept.

        Returns:
            Their id_ values, in order.
        """
        if not records:
            return []

        await self._collection.create_collection()
        latest = {document.id_: document for document in records}
        held = await self._held(list(latest))
        await self._replace(list(latest.values()), held)
        return [document.id_ for document in records]

    async def delete_many(self, ids: list[str]) -> list[str]:
        """Delete documents.

        Args:
            ids: The documents' id_ values.

        Returns:
            The id_ values deleted.

        Raises:
            RecordNotFoundError: When the collection holds no document of an id_.
        """
        held = await self._held(ids)
        missing = [id_ for id_ in ids if id_ not in held]

        if missing:
            msg = f"no document {', '.join(missing)}"
            raise RecordNotFoundError(msg)

        await self._reads.qdrant_client.delete(
            collection_name=self._reads.collection_name,
            points_selector=PointIdsList(points=[self._reads.point_id(id_) for id_ in ids]),
        )
        return ids

    async def delete_by_filters(self, filters: list[Filter]) -> int:
        """Delete every document the filters hold for, however many pages they fill.

        Args:
            filters: The conditions, all of which must hold.

        Returns:
            The number of documents deleted.

        Raises:
            ValueError: When no filter is given, which would match every document.
        """
        qdrant_filter = self._reads.qdrant_filter(filters)

        if qdrant_filter is None:
            msg = "delete_by_filters needs a filter; delete_all empties the collection"
            raise ValueError(msg)

        # Deleted by the filter, not by reading the matches first: a read stops at one page.
        matching = await self._reads.count_by_filters(filters)

        if matching:
            await self._reads.qdrant_client.delete(
                collection_name=self._reads.collection_name,
                points_selector=FilterSelector(filter=qdrant_filter),
            )

        return matching

    async def delete_all(self) -> int:
        """Delete every document of the collection.

        Returns:
            The number of documents deleted.
        """
        held = await self._reads.count_by_filters([])

        if held:
            await self._reads.qdrant_client.delete(
                collection_name=self._reads.collection_name,
                points_selector=FilterSelector(filter=QdrantFilter()),
            )

        return held

    async def _held(self, ids: list[str]) -> dict[str, T]:
        """Read the documents the collection holds among some id_ values.

        Args:
            ids: The id_ values.

        Returns:
            Each document held, by its id_.
        """
        reads = self._reads

        if not ids or not await reads.qdrant_client.collection_exists(reads.collection_name):
            return {}

        points = await reads.qdrant_client.retrieve(
            collection_name=reads.collection_name,
            ids=[reads.point_id(id_) for id_ in ids],
            with_payload=True,
        )
        documents = [reads.validated(reads.content_type, point.payload) for point in points]
        return {document.id_: document for document in documents}

    async def _replace(self, documents: Sequence[T], held: dict[str, T]) -> None:
        """Write documents, embedding only those new or whose text to embed changed.

        Args:
            documents: The documents, as they now are, one per id_.
            held: The documents the collection holds, by id_.
        """
        changed = [
            document
            for document in documents
            if document.id_ not in held
            or held[document.id_].text_to_embed != document.text_to_embed
        ]
        changed_ids = {document.id_ for document in changed}
        unchanged = [document for document in documents if document.id_ not in changed_ids]
        await self._upsert_points(changed)

        # The payload alone is replaced: the vectors stay as they were.
        await asyncio.gather(*[
            self._reads.qdrant_client.overwrite_payload(
                collection_name=self._reads.collection_name,
                payload=self._payload(document),
                points=[self._reads.point_id(document.id_)],
            )
            for document in unchanged
        ])

    async def _upsert_points(self, documents: Sequence[T]) -> None:
        """Embed documents by batches and write them as points, vectors and payload.

        Args:
            documents: The documents.
        """
        config = self._reads.config

        for start in range(0, len(documents), config.batch_size):
            batch = documents[start : start + config.batch_size]
            texts = [document.text_to_embed for document in batch]
            embedding = await self._reads.embedding_model.embed_documents(texts)
            points = []

            for document, text, dense in zip(batch, texts, embedding.embeddings, strict=True):
                vectors: dict[str, Any] = {QdrantReadRepository.DENSE: list(dense)}

                if config.use_sparse_vectors:
                    vectors[QdrantReadRepository.SPARSE] = Document(
                        text=text, model=config.sparse_model_name
                    )

                points.append(
                    PointStruct(
                        id=self._reads.point_id(document.id_),
                        payload=self._payload(document),
                        vector=vectors,
                    )
                )

            await self._reads.qdrant_client.upsert(
                collection_name=self._reads.collection_name, points=points
            )
            logger.debug(
                "Wrote documents {} to {} of {}", start + 1, start + len(batch), len(documents)
            )

    def _payload(self, document: T) -> dict[str, Any]:
        """Turn a document into a point payload, as JSON so a datetime index can read it.

        Args:
            document: The document.

        Returns:
            Its fields by name, its vector left out.
        """
        return document.model_dump(mode="json", exclude=self._unstored)
