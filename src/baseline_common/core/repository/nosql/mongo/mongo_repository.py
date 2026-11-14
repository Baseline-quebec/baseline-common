"""MongoRepository — reads and writes records of a MongoDB collection, through one object."""

from typing import Any

from pydantic import BaseModel
from pymongo.asynchronous.collection import AsyncCollection

from baseline_common.core.repository.abstract_repository import AbstractRepository
from baseline_common.core.repository.nosql.mongo.mongo_read_repository import MongoReadRepository
from baseline_common.core.repository.nosql.mongo.mongo_write_repository import MongoWriteRepository


class MongoRepository[T: BaseModel](
    MongoWriteRepository[T], MongoReadRepository[T], AbstractRepository[T]
):
    """Reads and writes one record type in its MongoDB collection; its id is the indexed _id."""

    def __init__(
        self,
        content_type: type[T],
        collection: AsyncCollection[dict[str, Any]],
        *,
        primary_key: str = "id",
    ) -> None:
        """Bind the collection of the record type, read and written by this same object.

        Args:
            content_type: The model every record is read as.
            collection: The collection holding the records, from a pymongo AsyncMongoClient.
            primary_key: The model's text field holding a record's id, stored as _id; the
                reader refuses a model without it.
        """
        MongoReadRepository.__init__(self, content_type, collection, primary_key=primary_key)
        MongoWriteRepository.__init__(self, self)
