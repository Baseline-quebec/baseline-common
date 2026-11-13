"""PostgresWriteRepository — writes one table; each call in its savepoint, the caller commits."""

import re
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import ClassVar, Self

from sqlalchemy import delete, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import DBAPIError
from sqlmodel import SQLModel

from baseline_common.core.repository.abstract_repository import (
    AbstractWriteRepository,
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_common.core.repository.sql.postgres.postgres_read_repository import (
    PostgresReadRepository,
)


class RejectionReason(StrEnum):
    """Which kind of constraint refused the write."""

    UNIQUE = "unique"
    UNKNOWN_REFERENCE = "unknown_reference"
    STILL_REFERENCED = "still_referenced"
    CHECK = "check"
    REQUIRED = "required"
    INVALID_VALUE = "invalid_value"


class RecordRejectedError(ValueError):
    """A write the database refused, with the constraint, columns and values it named."""

    SQLSTATE_REASONS: ClassVar[dict[str, RejectionReason]] = {
        "23505": RejectionReason.UNIQUE,
        "23514": RejectionReason.CHECK,
        "23502": RejectionReason.REQUIRED,
    }
    KEY_DETAIL: ClassVar[re.Pattern[str]] = re.compile(
        r"Key \((?P<columns>[^)]*)\)=\((?P<values>.*)\)"
    )
    TABLE_DETAIL: ClassVar[re.Pattern[str]] = re.compile(r'table "(?P<table>\w+)"')

    def __init__(
        self,
        reason: RejectionReason,
        table: str | None,
        constraint: str | None,
        columns: list[str],
        values: list[str],
        other_table: str | None,
    ) -> None:
        """Bind what the database said about the refusal.

        Args:
            reason: The kind of constraint.
            table: The table written to.
            constraint: The constraint's name.
            columns: The columns the constraint names, or the column a value was missing in.
            values: The values of those columns, when the database gave them.
            other_table: The table a foreign key points to, or is pointed at from.
        """
        super().__init__(f"{reason} on {table}: {constraint}")
        self.reason = reason
        self.table = table
        self.constraint = constraint
        self.columns = columns
        self.values = values
        self.other_table = other_table

    @classmethod
    def from_database(cls, error: DBAPIError) -> Self:
        """Read the refusal from the asyncpg error SQLAlchemy wraps.

        Args:
            error: The error raised at flush.

        Returns:
            The refusal.
        """
        cause = getattr(error.orig, "__cause__", None)
        sqlstate = str(getattr(cause, "sqlstate", "") or "")
        detail = str(getattr(cause, "detail", "") or "")
        key = cls.KEY_DETAIL.search(detail)
        other = cls.TABLE_DETAIL.search(detail)
        column = getattr(cause, "column_name", None)
        columns = [name.strip() for name in key["columns"].split(",")] if key else []

        if sqlstate == "23503":
            referenced = "still referenced" in detail
            reason = (
                RejectionReason.STILL_REFERENCED
                if referenced
                else RejectionReason.UNKNOWN_REFERENCE
            )
        else:
            reason = cls.SQLSTATE_REASONS.get(sqlstate, RejectionReason.INVALID_VALUE)

        return cls(
            reason=reason,
            table=getattr(cause, "table_name", None),
            constraint=getattr(cause, "constraint_name", None),
            columns=columns or ([column] if column else []),
            values=key["values"].split(", ") if key else [],
            other_table=other["table"] if other else None,
        )


class PostgresWriteRepository[T: SQLModel](AbstractWriteRepository[T]):
    """Writes one table; a call the database refuses raises ``RecordRejectedError`` and undoes."""

    def __init__(self, reads: PostgresReadRepository[T]) -> None:
        """Bind the reader of the same table; its session, id column and scope are used.

        Args:
            reads: Reads what a write checks.
        """
        self._reads = reads

    async def insert_many(self, records: Sequence[T]) -> list[str]:
        """Insert rows, put in the scope.

        Args:
            records: The rows.

        Returns:
            Their ids, in order.
        """
        scoped = [self._scoped(record) for record in records]
        ids = [self._id_of(record) for record in scoped]
        repeated = {id_ for id_ in ids if ids.count(id_) > 1} | await self._held_ids(ids)

        if repeated:
            raise DuplicateRecordError(", ".join(sorted(repeated)))

        async with self._atomically():
            self._reads.session.add_all(scoped)

        return [self._id_of(record) for record in scoped]

    async def update_many(self, records: Sequence[T]) -> list[str]:
        """Replace held rows with the ones of the same ids, owned children included.

        Args:
            records: The rows, as they now are.

        Returns:
            Their ids, in order.
        """
        await self._held([self._id_of(record) for record in records])
        return await self.upsert_many(records)

    async def upsert_many(self, records: Sequence[T]) -> list[str]:
        """Insert rows, each replacing the row of the same id, owned children included.

        Args:
            records: The rows, as they now are.

        Returns:
            Their ids, in order.
        """
        scoped = [self._scoped(record) for record in records]

        async with self._atomically():
            for record in scoped:
                self._leave_to_the_database(record)
                await self._reads.session.merge(record)

        return [self._id_of(record) for record in scoped]

    async def delete_many(self, ids: list[str]) -> list[str]:
        """Delete rows of the scope; the database applies its ``ON DELETE`` rules.

        Args:
            ids: The rows' ids.

        Returns:
            The ids deleted, in order.
        """
        await self._held(ids)
        statement = delete(self._reads.content_type).where(
            self._reads.key_condition(ids), self._reads.in_scope()
        )

        async with self._atomically():
            await self._reads.session.execute(statement)

        return ids

    @asynccontextmanager
    async def _atomically(self) -> AsyncGenerator[None]:
        """Run a write in a savepoint and flush it, so a refusal undoes only this call."""
        session = self._reads.session

        try:
            async with session.begin_nested():
                yield
                await session.flush()

        except DBAPIError as error:
            rejection = RecordRejectedError.from_database(error)
            primary_key = [
                column.name for column in sa_inspect(self._reads.content_type).primary_key
            ]

            if rejection.reason is RejectionReason.UNIQUE and rejection.columns == primary_key:
                raise DuplicateRecordError(", ".join(rejection.values)) from error

            raise rejection from error

    async def _held(self, ids: list[str]) -> None:
        """Raise ``RecordNotFoundError`` unless every id is held in the scope."""
        missing = set(ids) - await self._held_ids(ids)

        if missing:
            raise RecordNotFoundError(", ".join(sorted(missing)))

    async def _held_ids(self, ids: list[str]) -> set[str]:
        """The ids among these that the table holds, in any scope: a key is unique table-wide."""
        if not ids:
            return set()

        column = self._reads.column(self._reads.id_field)
        statement = select(column).where(self._reads.key_condition(ids))
        return {str(id_) for id_ in (await self._reads.session.execute(statement)).scalars()}

    def _leave_to_the_database(self, record: SQLModel) -> None:
        """Unset the empty columns the database fills, here and in owned children, so merge keeps them."""
        state = sa_inspect(record)

        for column in state.mapper.columns:
            if column.server_default is not None and state.dict.get(column.key) is None:
                state.dict.pop(column.key, None)

        for relationship in state.mapper.relationships:
            children = state.dict.get(relationship.key) or []

            for child in children if isinstance(children, list) else [children]:
                self._leave_to_the_database(child)

    def _scoped(self, record: T) -> T:
        """The record with the scope's values set."""
        for name, value in self._reads.scope.items():
            setattr(record, name, value)

        return record

    def _id_of(self, record: T) -> str:
        """The record's id, as text."""
        return str(getattr(record, self._reads.id_field))
