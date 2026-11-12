"""MongoReadRepository — reads records from a MongoDB collection, by id or by filters."""

import re
from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel
from pymongo import ASCENDING, DESCENDING
from pymongo.asynchronous.collection import AsyncCollection

from baseline_common.core.repository.abstract_repository import AbstractReadRepository
from baseline_common.core.repository.query_filters import (
    ComparisonFilter,
    Filter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)


class MongoReadRepository[T: BaseModel](AbstractReadRepository[T]):
    """Reads one record type from its MongoDB collection; its id is the indexed _id."""

    OPERATORS: ClassVar[dict[str, str]] = {
        "==": "$eq",
        "!=": "$ne",
        ">": "$gt",
        "<": "$lt",
        ">=": "$gte",
        "<=": "$lte",
    }

    def __init__(
        self,
        content_type: type[T],
        collection: AsyncCollection[dict[str, Any]],
        *,
        primary_key: str = "id",
    ) -> None:
        """Bind the collection of the record type.

        Args:
            content_type: The model every record is read as.
            collection: The collection holding the records, from a pymongo AsyncMongoClient.
            primary_key: The model's text field holding a record's id, stored as _id.

        Raises:
            ValueError: When the model has no such top-level text field.
        """
        super().__init__(content_type)
        field = content_type.model_fields.get(primary_key)

        if field is None or field.annotation not in {str, str | None}:
            msg = f"{content_type.__name__} has no text field {primary_key!r} to hold its id"
            raise ValueError(msg)

        self.collection = collection
        self.primary_key = primary_key

    async def get_by_id(self, id_: str) -> T | None:
        """Find a record by its id.

        Args:
            id_: The record's id.

        Returns:
            The record, or None when there is none.
        """
        document = await self.collection.find_one({"_id": id_})
        return None if document is None else self._record(document)

    async def get_by_filters(
        self,
        filters: Sequence[Filter] = (),
        sort_by: str | None = None,
        sort_order: SortOrder = SortOrder.ASC,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[T]:
        """Find the records every filter holds for.

        Args:
            filters: The conditions, all of which must hold.
            sort_by: The model field path to sort on; a missing value sorts lowest, as in MongoDB.
            sort_order: The order to sort in; equal values, or no sort_by, follow the id.
            limit: The most records returned; None returns them all.
            offset: How many matching records to skip.

        Returns:
            The matching records.
        """
        if limit is not None and limit <= 0:
            return []

        # Equal values come back in any order, so a page could repeat or skip a record: the id
        # breaks the tie. Azure Cosmos DB sorts on two fields only with a compound index on both.
        order = [("_id", ASCENDING)]

        if sort_by:
            direction = ASCENDING if sort_order is SortOrder.ASC else DESCENDING
            order.insert(0, (self.stored_path(sort_by), direction))

        cursor = self.collection.find(
            self.mongo_query(filters), sort=order, skip=offset, limit=limit or 0
        )
        return [self._record(document) async for document in cursor]

    async def count_by_filters(self, filters: list[Filter]) -> int:
        """Count the records every filter holds for.

        Args:
            filters: The conditions, all of which must hold.

        Returns:
            The number of matching records.
        """
        return await self.collection.count_documents(self.mongo_query(filters))

    def mongo_query(self, filters: Sequence[Filter]) -> dict[str, Any]:
        """Combine conditions into the MongoDB query all of them must hold for.

        Args:
            filters: The conditions.

        Returns:
            The query; an empty one matches every document.
        """
        conditions = [self._condition(condition) for condition in filters]
        return {"$and": conditions} if conditions else {}

    def stored_path(self, path: str) -> str:
        """Name a model field path as the document stores it.

        Args:
            path: The model field path.

        Returns:
            _id for the primary key, the path itself otherwise.
        """
        return "_id" if path == self.primary_key else path

    def _condition(self, condition: Filter) -> dict[str, Any]:
        """Translate a condition into a MongoDB query.

        Args:
            condition: The condition.

        Returns:
            The query.

        Raises:
            ValueError: When MongoDB has no counterpart for the condition.
        """
        if isinstance(condition, LogicalFilter):
            operands = [self._condition(operand) for operand in condition.operands]

            # $and and $or refuse an empty list: no operand holds for all, none for any.
            if not operands:
                return {} if condition.operator == "&" else {"_id": {"$in": []}}

            return {"$and" if condition.operator == "&" else "$or": operands}

        if isinstance(condition, ComparisonFilter):
            # Exact, case included: ignoring case needs a collation, which Cosmos DB lacks.
            operator = self.OPERATORS[condition.operator]
            return {self.stored_path(condition.field): {operator: condition.value}}

        if isinstance(condition, TextFilter):
            # No index serves a case-insensitive regex: the collection is scanned.
            pattern = {"$regex": re.escape(condition.value), "$options": "i"}
            return {self.stored_path(condition.field): pattern}

        msg = f"MongoDB cannot apply a {type(condition).__name__} ({condition.operator})"
        raise ValueError(msg)

    def _record(self, document: dict[str, Any]) -> T:
        """Read a document as a record, its _id back in the primary key.

        Args:
            document: The document.

        Returns:
            The record; a document that does not fit raises, never skipped.
        """
        fields = dict(document)
        fields[self.primary_key] = fields.pop("_id")
        return self.content_type.model_validate(fields, by_name=True, by_alias=False)
