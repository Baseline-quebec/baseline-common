"""SqlWriteRepository — writes one table; each call in its savepoint, the caller commits."""

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from enum import StrEnum
from itertools import batched
from typing import ClassVar, Self, cast

from sqlalchemy import Table, delete, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import DBAPIError
from sqlmodel import SQLModel

from baseline_common.core.repository.abstract_repository import (
    AbstractWriteRepository,
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_common.core.repository.sql.abstract_rejection_translator import (
    AbstractRejectionTranslator,
    RejectionReason,
)
from baseline_common.core.repository.sql.generic_rejection_translator import (
    GenericRejectionTranslator,
)
from baseline_common.core.repository.sql.postgres_rejection_translator import (
    PostgresRejectionTranslator,
)
from baseline_common.core.repository.sql.sql_read_repository import SqlReadRepository
from baseline_common.core.repository.sql.sqlserver_rejection_translator import (
    SqlServerRejectionTranslator,
)


class SqlDialect(StrEnum):
    """The SQLAlchemy dialects whose refusals a translator reads; any other dialect is ``OTHER``."""

    POSTGRESQL = "postgresql"
    MSSQL = "mssql"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> Self:  # noqa: ARG003
        """Take a dialect no member names as ``OTHER``.

        Args:
            value: The dialect's name.

        Returns:
            ``OTHER``.
        """
        return cls.OTHER


class SqlWriteRepository[T: SQLModel](AbstractWriteRepository[T]):
    """Writes one table; a call the database refuses raises ``RecordRejectedError`` and undoes."""

    # SQL Server takes at most 2100 parameters in a statement.
    KEYS_PER_STATEMENT: ClassVar[int] = 1000

    TRANSLATORS: ClassVar[dict[SqlDialect, type[AbstractRejectionTranslator]]] = {
        SqlDialect.POSTGRESQL: PostgresRejectionTranslator,
        SqlDialect.MSSQL: SqlServerRejectionTranslator,
        SqlDialect.OTHER: GenericRejectionTranslator,
    }

    def __init__(self, reads: SqlReadRepository[T]) -> None:
        """Bind the reader of the same table, and the translator of its session's dialect.

        Args:
            reads: Reads what a write checks; its session, id column and scope are used.
        """
        self._reads = reads
        dialect = SqlDialect(reads.session.get_bind().dialect.name)
        table = cast("Table", sa_inspect(reads.content_type).local_table)
        self._translator = self.TRANSLATORS[dialect](table)

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

        async with self._atomically():
            for batch in batched(ids, self.KEYS_PER_STATEMENT):
                statement = delete(self._reads.content_type).where(
                    self._reads.key_condition(batch), self._reads.in_scope()
                )
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
            # A deadlock or a lost connection is no refusal: it stays itself.
            if not self._translator.is_refusal(error):
                raise

            rejection = self._translator.translate(error)
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
        column = self._reads.column(self._reads.id_field)
        held: set[str] = set()

        for batch in batched(ids, self.KEYS_PER_STATEMENT):
            statement = select(column).where(self._reads.key_condition(batch))
            held |= {str(id_) for id_ in (await self._reads.session.execute(statement)).scalars()}

        return held

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
