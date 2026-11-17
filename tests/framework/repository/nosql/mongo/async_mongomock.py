"""An async front on mongomock."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

from pymongo import ReplaceOne
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.results import DeleteResult, InsertManyResult


class MongomockCursor:
    """A mongomock cursor read with ``async for``, as a pymongo AsyncCursor is."""

    def __init__(self, cursor: Cursor[dict[str, Any]]) -> None:
        """Wrap the cursor.

        Args:
            cursor: The mongomock cursor.
        """
        self._cursor = cursor

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """Yield the documents.

        Yields:
            Each document.
        """
        for document in self._cursor:
            yield document


class AsyncMongomock:
    """A mongomock collection behind the async calls of a pymongo AsyncCollection."""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        """Wrap the collection.

        Args:
            collection: The mongomock collection.
        """
        self.name = collection.name
        """The collection's name."""
        self.collection = collection

    def find(self, query: dict[str, Any], **options: object) -> MongomockCursor:
        """Find documents.

        Args:
            query: The query.
            options: The sort, skip, limit or projection.

        Returns:
            Their cursor.
        """
        return MongomockCursor(self.collection.find(query, **options))

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        """Find a document.

        Args:
            query: The query.

        Returns:
            The document, or None.
        """
        return self.collection.find_one(query)

    async def count_documents(self, query: dict[str, Any]) -> int:
        """Count documents.

        Args:
            query: The query.

        Returns:
            The count.
        """
        return self.collection.count_documents(query)

    async def insert_many(self, documents: list[dict[str, Any]]) -> InsertManyResult:
        """Insert documents, in order.

        Args:
            documents: The documents.

        Returns:
            The result.
        """
        return self.collection.insert_many(documents)

    async def bulk_write(self, requests: Sequence[ReplaceOne[dict[str, Any]]]) -> None:
        """Replay a bulk of replacements one by one: mongomock 4.3 takes no pymongo 4.13+ one.

        Args:
            requests: The replacements.
        """
        for request in requests:
            self.collection.replace_one(
                request._filter, request._doc, upsert=bool(request._upsert)
            )

    async def delete_one(self, query: dict[str, Any]) -> DeleteResult:
        """Delete a document.

        Args:
            query: The query.

        Returns:
            The result.
        """
        return self.collection.delete_one(query)

    async def delete_many(self, query: dict[str, Any]) -> DeleteResult:
        """Delete documents.

        Args:
            query: The query.

        Returns:
            The result.
        """
        return self.collection.delete_many(query)
