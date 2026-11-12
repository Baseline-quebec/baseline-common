"""MongoWriteRepository — writes records to a MongoDB collection, its _id refusing duplicates."""

from collections import Counter
from collections.abc import Sequence
from typing import Any, ClassVar

from bson import ObjectId
from loguru import logger
from pydantic import BaseModel
from pymongo import ReplaceOne
from pymongo.errors import BulkWriteError

from baseline_common.core.repository.abstract_repository import (
    AbstractWriteRepository,
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_common.core.repository.nosql.mongo.mongo_read_repository import MongoReadRepository
from baseline_common.core.repository.query_filters import Filter


class MongoWriteRepository[T: BaseModel](AbstractWriteRepository[T]):
    """Writes one record type to its MongoDB collection, created on the first write."""

    DUPLICATE_KEY: ClassVar[int] = 11000

    def __init__(self, reads: MongoReadRepository[T]) -> None:
        """Bind the reader of the same record type.

        Args:
            reads: Reads what a write checks; its collection and primary key are used.
        """
        self._reads = reads

    async def insert_many(self, records: Sequence[T]) -> list[str]:
        """Insert records, giving one without an id a fresh one.

        Args:
            records: The records.

        Returns:
            Their ids, in order.

        Raises:
            DuplicateRecordError: When an id is already held, or given twice.
        """
        records = [self._identified(record) for record in records]
        ids = [self._id_of(record) for record in records]

        if not ids:
            return []

        held = await self._held(ids)
        repeated = {id_ for id_, times in Counter(ids).items() if times > 1} | held

        if repeated:
            msg = f"already held: {', '.join(sorted(repeated))}"
            raise DuplicateRecordError(msg)

        try:
            await self._reads.collection.insert_many([self._document(r) for r in records])
        except BulkWriteError as error:
            # Another writer took an id since the check; the records before it are written.
            taken = [
                e["op"]["_id"]
                for e in error.details["writeErrors"]
                if e["code"] == self.DUPLICATE_KEY
            ]

            if not taken:
                raise

            msg = f"already held: {', '.join(taken)}"
            raise DuplicateRecordError(msg) from error

        logger.info("Inserted {} records into {}", len(ids), self._reads.collection.name)
        return ids

    async def update_many(self, records: Sequence[T]) -> list[str]:
        """Replace records with the ones of the same ids.

        Args:
            records: The records, as they now are.

        Returns:
            Their ids, in order.

        Raises:
            RecordNotFoundError: When the collection holds no record of an id; none is written.
        """
        ids = [self._id_of(record) for record in records]
        held = await self._held(ids)
        missing = [id_ for id_ in ids if id_ not in held]

        if missing:
            msg = f"no record {', '.join(missing)}"
            raise RecordNotFoundError(msg)

        await self._replace(records, upsert=False)
        return ids

    async def upsert_many(self, records: Sequence[T]) -> list[str]:
        """Insert records, or replace those of the same ids.

        Args:
            records: The records, as they now are; the last of an id repeated is kept.

        Returns:
            Their ids, in order.
        """
        records = [self._identified(record) for record in records]
        latest = {self._id_of(record): record for record in records}
        await self._replace(list(latest.values()), upsert=True)
        logger.info("Upserted {} records into {}", len(latest), self._reads.collection.name)
        return [self._id_of(record) for record in records]

    async def delete_many(self, ids: list[str]) -> list[str]:
        """Delete records, none of them when one is not held.

        Args:
            ids: The records' ids.

        Returns:
            The ids deleted.

        Raises:
            RecordNotFoundError: When the collection holds no record of an id.
        """
        held = await self._held(ids)
        missing = [id_ for id_ in ids if id_ not in held]

        if missing:
            msg = f"no record {', '.join(missing)}"
            raise RecordNotFoundError(msg)

        await self._reads.collection.delete_many({"_id": {"$in": ids}})
        return ids

    async def delete_by_filters(self, filters: list[Filter]) -> int:
        """Delete every record the filters hold for.

        Args:
            filters: The conditions, all of which must hold.

        Returns:
            The number of records deleted.

        Raises:
            ValueError: When no filter is given, which would match every record.
        """
        if not filters:
            msg = "delete_by_filters needs a filter; delete_all empties the collection"
            raise ValueError(msg)

        result = await self._reads.collection.delete_many(self._reads.mongo_query(filters))
        logger.info(
            "Deleted {} records from {}", result.deleted_count, self._reads.collection.name
        )
        return result.deleted_count

    async def delete_all(self) -> int:
        """Delete every record of the collection.

        Returns:
            The number of records deleted.
        """
        result = await self._reads.collection.delete_many({})
        logger.info(
            "Deleted {} records from {}", result.deleted_count, self._reads.collection.name
        )
        return result.deleted_count

    async def _held(self, ids: list[str]) -> set[str]:
        """Find which ids the collection holds, by its _id index.

        Args:
            ids: The ids.

        Returns:
            Those held.
        """
        if not ids:
            return set()

        cursor = self._reads.collection.find({"_id": {"$in": ids}}, projection=["_id"])
        return {document["_id"] async for document in cursor}

    async def _replace(self, records: Sequence[T], *, upsert: bool) -> None:
        """Replace the documents of records in one round trip, inserting those absent if asked.

        Args:
            records: The records, one per id.
            upsert: Whether a record absent from the collection is inserted.
        """
        # pymongo refuses an empty bulk.
        if not records:
            return

        await self._reads.collection.bulk_write([
            ReplaceOne({"_id": self._id_of(record)}, self._document(record), upsert=upsert)
            for record in records
        ])

    def _identified(self, record: T) -> T:
        """Give a record its id when it has none yet.

        Args:
            record: The record.

        Returns:
            The record, or a copy of it with a fresh ObjectId as text, which sorts by creation.
        """
        if getattr(record, self._reads.primary_key) is not None:
            return record

        return record.model_copy(update={self._reads.primary_key: str(ObjectId())})

    def _id_of(self, record: T) -> str:
        """Read a record's id.

        Args:
            record: The record.

        Returns:
            The value of its primary key, as text.
        """
        return str(getattr(record, self._reads.primary_key))

    def _document(self, record: T) -> dict[str, Any]:
        """Turn a record into its document, its primary key moved to _id.

        Args:
            record: The record.

        Returns:
            Its fields by name, as JSON.
        """
        # As JSON, like a filter's value it is compared with: a datetime is ISO text.
        document = record.model_dump(mode="json")
        document["_id"] = document.pop(self._reads.primary_key)
        return document
