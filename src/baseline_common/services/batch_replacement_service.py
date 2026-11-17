"""BatchReplacementService — writes a batch in place of the records held."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from baseline_common.framework.repository.abstract_repository import AbstractRepository


class BatchReplacement(BaseModel):
    """The ids a replacement updated, inserted and deleted."""

    model_config = ConfigDict(frozen=True)

    updated: list[str]
    """The ids reused: records rewritten in place of the ones held."""
    inserted: list[str]
    """The ids of the records the batch had beyond those held."""
    deleted: list[str]
    """The ids of the held records the batch no longer has."""


class BatchReplacementService[T: BaseModel]:
    """Replace the records held with a batch: reuse their ids, insert the extra, delete the rest."""

    def __init__(self, repository: AbstractRepository[T]) -> None:
        """Bind the records held; a scoped repository (``SqlRepository``) limits the batch.

        Args:
            repository: Holds the previous batch, and takes the new one in its place.
        """
        self.repository = repository

    async def replace(self, records: Sequence[T]) -> BatchReplacement:
        """Write a batch in place of the records held, reusing their ids in order.

        Args:
            records: The batch, without ids.

        Returns:
            The ids updated, inserted and deleted.
        """
        primary_key = self.repository.primary_key
        previous = await self.repository.get_by_filters(sort_by=primary_key, limit=None)

        kept = [
            record.model_copy(update={primary_key: getattr(held, primary_key)})
            for record, held in zip(records, previous, strict=False)
        ]
        extra = records[len(previous) :]
        leftover = [getattr(held, primary_key) for held in previous[len(records) :]]

        updated = await self.repository.update_many(kept) if kept else []
        inserted = await self.repository.insert_many(extra) if extra else []
        deleted = await self.repository.delete_many(leftover) if leftover else []

        return BatchReplacement(updated=updated, inserted=inserted, deleted=deleted)
