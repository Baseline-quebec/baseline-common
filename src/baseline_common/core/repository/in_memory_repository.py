"""InMemoryRepository — records held in memory, each filter matched against them."""

from collections.abc import Iterable, Sequence
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from baseline_common.core.repository.abstract_repository import (
    AbstractRepository,
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_common.core.repository.query_filters import Filter, SortOrder


class InMemoryRepository[T: BaseModel](AbstractRepository[T]):
    """Reads and writes records held in memory, for loaded files and tests."""

    def __init__(
        self, content_type: type[T], records: Iterable[T] = (), *, primary_key: str = "id"
    ) -> None:
        """Hold the records.

        Args:
            content_type: The model every record is.
            records: The records.
            primary_key: The model field path holding a record's primary key.
        """
        super().__init__(content_type)
        self._records = list(records)
        self._primary_key = primary_key

    async def get_by_id(self, id_: str) -> T | None:
        """Find a record by its id.

        Args:
            id_: The value of the record's id field.

        Returns:
            The record, or None.
        """
        return next((r for r in self._records if self._id_of(r) == id_), None)

    async def insert_many(self, records: Sequence[T]) -> list[str]:
        """Add records, giving one without an id a fresh one, as a backend would.

        Args:
            records: The records.

        Returns:
            Their ids.

        Raises:
            DuplicateRecordError: When an id is already held, or given twice.
        """
        records = [self._identified(record) for record in records]
        ids = [self._id_of(record) for record in records]
        held = [self._id_of(record) for record in self._records]
        repeated = sorted({id_ for id_ in ids if id_ in held or ids.count(id_) > 1})

        if repeated:
            msg = f"already held: {', '.join(repeated)}"
            raise DuplicateRecordError(msg)

        self._records.extend(records)
        return ids

    async def update_many(self, records: Sequence[T]) -> list[str]:
        """Replace records with the ones of the same ids.

        Args:
            records: The records, as they now are.

        Returns:
            Their ids.
        """
        ids = [self._id_of(record) for record in records]
        positions = [self._position(id_) for id_ in ids]

        for position, record in zip(positions, records, strict=True):
            self._records[position] = record

        return ids

    async def upsert_many(self, records: Sequence[T]) -> list[str]:
        """Add records, each in place of every held record of its id.

        Args:
            records: The records; one without an id is given one.

        Returns:
            Their ids.
        """
        records = [self._identified(record) for record in records]
        latest = {self._id_of(record): record for record in records}
        self._records = [r for r in self._records if self._id_of(r) not in latest]
        self._records.extend(latest.values())
        return [self._id_of(record) for record in records]

    async def delete_many(self, ids: list[str]) -> list[str]:
        """Remove records, none of them when one is not held.

        Args:
            ids: The records' ids.

        Returns:
            The ids removed.
        """
        positions = {self._position(id_) for id_ in ids}
        self._records = [r for at, r in enumerate(self._records) if at not in positions]
        return ids

    def _position(self, id_: str) -> int:
        """Find where a record is held.

        Args:
            id_: The record's id.

        Returns:
            Its position.

        Raises:
            RecordNotFoundError: When no record has that id.
        """
        position = next(
            (at for at, record in enumerate(self._records) if self._id_of(record) == id_), None
        )

        if position is None:
            msg = f"no record {id_}"
            raise RecordNotFoundError(msg)

        return position

    def _identified(self, record: T) -> T:
        """Give a record its id when it has none yet.

        Args:
            record: The record.

        Returns:
            The record, or a copy of it with a fresh id.
        """
        if Filter.value_of(record, self._primary_key) is not None:
            return record

        return record.model_copy(update={self._primary_key: uuid4().hex})

    def _id_of(self, record: T) -> str:
        """Read a record's id.

        Args:
            record: The record.

        Returns:
            The value of its id field, as text.
        """
        return str(Filter.value_of(record, self._primary_key))

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
            sort_by: The model field path to sort on, missing values last.
            sort_order: The order to sort in.
            limit: The most records returned; None returns them all.
            offset: How many matching records to skip.

        Returns:
            The matching records.
        """
        found = [r for r in self._records if all(f.matches(r) for f in filters)]

        if sort_by:
            found.sort(
                key=lambda record: self._sort_key(Filter.value_of(record, sort_by)),
                reverse=sort_order is SortOrder.DESC,
            )

        return found[offset : None if limit is None else offset + limit]

    @staticmethod
    def _sort_key(value: Any) -> tuple[bool, Any]:  # noqa: ANN401
        """Sort missing values last, and text regardless of case.

        Args:
            value: A field's value.

        Returns:
            The key to sort on.
        """
        return (value is None, Filter.folded(value))
