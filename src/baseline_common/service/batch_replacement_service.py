"""BatchReplacementService — writes a batch in place of the previous batch of the same scope."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from baseline_common.core.repository.abstract_repository import (
    AbstractReadRepository,
    AbstractWriteRepository,
)
from baseline_common.core.repository.query_filters import EqualityFilter


class BatchReplacement(BaseModel):
    """The ids a replacement updated, inserted and deleted."""

    model_config = ConfigDict(frozen=True)

    updated: list[str]
    inserted: list[str]
    deleted: list[str]


class BatchReplacementService:
    """Rewrite a scope's records in place: reuse their ids, insert the extra, delete the rest."""

    def __init__(
        self,
        reads: AbstractReadRepository[Any],
        writes: AbstractWriteRepository[Any],
        scope: list[str],
        id_field: str,
    ) -> None:
        """Bind the backend and what makes two records part of the same batch.

        Args:
            reads: Reads the records held, by scope.
            writes: Writes them.
            scope: The model fields every record of a batch shares, e.g. the machine and the week.
            id_field: The model field holding a record's id in the backend.

        Raises:
            ValueError: When the scope names no field.
        """
        if not scope:
            msg = "scope must name at least one field"
            raise ValueError(msg)

        self.reads = reads
        self.writes = writes
        self.scope = scope
        self.id_field = id_field

    async def replace(self, records: Sequence[BaseModel]) -> BatchReplacement:
        """Write a batch in place of the records held under its scope.

        Args:
            records: The batch, all of one scope, without ids.

        Returns:
            The ids updated, inserted and deleted.
        """
        # Sorted on the id so a backend that pages by offset neither skips nor repeats one.
        previous = await self.reads.get_by_filters(
            self._scope_of(records), sort_by=self.id_field, limit=None
        )

        # Paired in id order, not by content: the scope keeps its ids, not each record its own.
        kept = [
            record.model_copy(update={self.id_field: getattr(held, self.id_field)})
            for record, held in zip(records, previous, strict=False)
        ]
        extra = records[len(previous) :]
        leftover = [getattr(held, self.id_field) for held in previous[len(records) :]]

        # The write contract does not promise that an empty write sends nothing.
        updated = await self.writes.update_many(kept) if kept else []
        inserted = await self.writes.insert_many(extra) if extra else []
        deleted = await self.writes.delete_many(leftover) if leftover else []

        return BatchReplacement(updated=updated, inserted=inserted, deleted=deleted)

    def _scope_of(self, records: Sequence[BaseModel]) -> list[EqualityFilter]:
        """Read the scope a batch shares, each value as the backend's filters compare it.

        Args:
            records: The batch.

        Returns:
            One filter per scope field, on the batch's value.

        Raises:
            ValueError: When the batch is empty, or its records differ on a scope field.
        """
        if not records:
            msg = "an empty batch names no scope: nothing tells which records it replaces"
            raise ValueError(msg)

        rows = [record.model_dump(mode="json", include=set(self.scope)) for record in records]
        differing = [field for field in self.scope if len({row[field] for row in rows}) > 1]

        if differing:
            msg = f"a batch shares its scope, yet its records differ on {', '.join(differing)}"
            raise ValueError(msg)

        return [EqualityFilter(field=field, value=rows[0][field]) for field in self.scope]
