"""The repository contracts every backend of a record type fulfills: reading, writing, or both."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel

from baseline_common.core.repository.query_filters import Filter, SortOrder


class AbstractReadRepository[T: BaseModel](ABC):
    """Reads the records of one type, whatever backend holds them."""

    def __init__(self, content_type: type[T]) -> None:
        """Bind the record type.

        Args:
            content_type: The model every record is read as.
        """
        self.content_type = content_type

    @abstractmethod
    async def get_by_id(self, id_: str) -> T | None:
        """Find a record by its id.

        Args:
            id_: The record's id in the backend.

        Returns:
            The record, or None when there is none.
        """

    @abstractmethod
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
            filters: The conditions, all of which must hold; none returns every record.
            sort_by: The model field path to sort on.
            sort_order: The order to sort in.
            limit: The most records returned; None returns them all.
            offset: How many matching records to skip.

        Returns:
            The matching records.
        """


class RecordNotFoundError(LookupError):
    """A read or a write names a record the backend does not hold."""


class DuplicateRecordError(ValueError):
    """An insert gives a record the id of one the backend already holds."""


class AbstractWriteRepository[T: BaseModel](ABC):
    """Writes the records of one type, whatever backend holds them."""

    async def insert_one(self, record: T) -> str:
        """Insert a record.

        Args:
            record: The record.

        Returns:
            Its id in the backend.
        """
        (id_,) = await self.insert_many([record])
        return id_

    async def update_one(self, record: T) -> str:
        """Replace a record with the one of the same id.

        Args:
            record: The record, as it now is.

        Returns:
            Its id in the backend.
        """
        (id_,) = await self.update_many([record])
        return id_

    async def upsert_one(self, record: T) -> str:
        """Insert a record, or replace the one of the same id.

        Args:
            record: The record, as it now is.

        Returns:
            Its id in the backend.
        """
        (id_,) = await self.upsert_many([record])
        return id_

    @abstractmethod
    async def insert_many(self, records: Sequence[T]) -> list[str]:
        """Insert records.

        Args:
            records: The records.

        Returns:
            Their ids in the backend, in order.
        """

    @abstractmethod
    async def update_many(self, records: Sequence[T]) -> list[str]:
        """Replace records with the ones of the same ids.

        Args:
            records: The records, as they now are.

        Returns:
            Their ids in the backend, in order.
        """

    @abstractmethod
    async def upsert_many(self, records: Sequence[T]) -> list[str]:
        """Insert records, each replacing whatever the backend holds under its id.

        Args:
            records: The records, as they now are; the last of an id repeated is kept.

        Returns:
            Their ids in the backend, in order.
        """

    async def delete_one(self, id_: str) -> str:
        """Delete a record.

        Args:
            id_: The record's id in the backend.

        Returns:
            The id deleted.
        """
        (deleted,) = await self.delete_many([id_])
        return deleted

    @abstractmethod
    async def delete_many(self, ids: list[str]) -> list[str]:
        """Delete records.

        Args:
            ids: The records' ids in the backend.

        Returns:
            The ids deleted, in order.
        """


class AbstractRepository[T: BaseModel](AbstractReadRepository[T], AbstractWriteRepository[T]):
    """Reads and writes the records of one type, whatever backend holds them."""
