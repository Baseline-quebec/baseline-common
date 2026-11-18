"""SqlWriteRepository — writes one table; each call in its savepoint, the caller commits."""

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from enum import StrEnum
from itertools import batched, starmap
from typing import Any, ClassVar, Self, cast

from sqlalchemy import ColumnElement, Table, and_, delete, exists, or_, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Mapper, RelationshipDirection, RelationshipProperty
from sqlmodel import SQLModel

from baseline_common.framework.repository.abstract_repository import (
    AbstractWriteRepository,
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_common.framework.repository.sql.abstract_rejection_translator import (
    AbstractRejectionTranslator,
    RejectionReason,
)
from baseline_common.framework.repository.sql.generic_rejection_translator import (
    GenericRejectionTranslator,
)
from baseline_common.framework.repository.sql.postgres_rejection_translator import (
    PostgresRejectionTranslator,
)
from baseline_common.framework.repository.sql.sql_read_repository import SqlReadRepository
from baseline_common.framework.repository.sql.sqlserver_rejection_translator import (
    SqlServerRejectionTranslator,
)


class SqlDialect(StrEnum):
    """The SQLAlchemy dialects whose refusals a translator reads; any other dialect is ``OTHER``."""

    POSTGRESQL = "postgresql"
    """PostgreSQL."""
    MSSQL = "mssql"
    """SQL Server."""
    OTHER = "other"
    """Any other dialect."""

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

    KEYS_PER_STATEMENT: ClassVar[int] = 1000
    """The most ids one statement names: SQL Server takes at most 2100 parameters in one."""

    TRANSLATORS: ClassVar[dict[SqlDialect, type[AbstractRejectionTranslator]]] = {
        SqlDialect.POSTGRESQL: PostgresRejectionTranslator,
        SqlDialect.MSSQL: SqlServerRejectionTranslator,
        SqlDialect.OTHER: GenericRejectionTranslator,
    }
    """The translator that reads each dialect's refusals."""

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
            msg = f"already held: {', '.join(sorted(repeated))}"
            raise DuplicateRecordError(msg)

        async with self._atomically():
            self._reads.session.add_all(scoped)

        return [self._id_of(record) for record in scoped]

    async def update_many(self, records: Sequence[T]) -> list[str]:
        """Replace rows of the scope with the ones of the same ids, owned children included.

        Args:
            records: The rows, as they now are.

        Returns:
            Their ids, in order.

        Raises:
            RecordNotFoundError: When the scope holds no row of an id; nothing is written.
            DuplicateRecordError: When an owned child's id is held outside the scope; nothing is
                written.
        """
        missing = await self._missing([self._id_of(record) for record in records])

        if missing:
            msg = f"no record {', '.join(missing)}"
            raise RecordNotFoundError(msg)

        elsewhere = await self._held_elsewhere(records)

        if elsewhere:
            msg = f"already held: {', '.join(elsewhere)}"
            raise DuplicateRecordError(msg)

        return await self._merged(records)

    async def upsert_many(self, records: Sequence[T]) -> list[str]:
        """Insert rows, each replacing the row of the same id, owned children included.

        Args:
            records: The rows, as they now are.

        Returns:
            Their ids, in order.

        Raises:
            DuplicateRecordError: When the key a row or an owned child is merged on is held
                outside the scope, so the upsert would move it; nothing is written.
        """
        elsewhere = await self._held_elsewhere(records)

        if elsewhere:
            msg = f"already held: {', '.join(elsewhere)}"
            raise DuplicateRecordError(msg)

        return await self._merged(records)

    async def delete_many(self, ids: list[str]) -> list[str]:
        """Delete rows of the scope; the database applies its ``ON DELETE`` rules.

        Args:
            ids: The rows' ids.

        Returns:
            The ids deleted, in order.

        Raises:
            RecordNotFoundError: When the scope holds no row of an id, though another scope may;
                nothing is deleted.
        """
        missing = await self._missing(ids)

        if missing:
            msg = f"no record {', '.join(missing)}"
            raise RecordNotFoundError(msg)

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
                msg = f"already held: {', '.join(rejection.values)}"
                raise DuplicateRecordError(msg) from error

            raise rejection from error

    async def _merged(self, records: Sequence[T]) -> list[str]:
        """Merge rows, put in the scope, each on the row of the same key."""
        scoped = [self._scoped(record) for record in records]

        async with self._atomically():
            for record in scoped:
                self._leave_to_the_database(record)
                await self._reads.session.merge(record)

        return [self._id_of(record) for record in scoped]

    async def _missing(self, ids: list[str]) -> list[str]:
        """The ids among these the scope does not hold, sorted."""
        return sorted(set(ids) - await self._held_ids(ids, self._reads.in_scope()))

    async def _held_ids(self, ids: list[str], *conditions: ColumnElement[bool]) -> set[str]:
        """The ids among these the table holds, in any scope unless the conditions narrow it."""
        column = self._reads.column(self._reads.primary_key)
        held: set[str] = set()

        for batch in batched(ids, self.KEYS_PER_STATEMENT):
            statement = select(column).where(self._reads.key_condition(batch), *conditions)
            held |= {str(id_) for id_ in (await self._reads.session.execute(statement)).scalars()}

        return held

    async def _held_elsewhere(self, records: Sequence[T]) -> list[str]:
        """The keys merge would match, of these rows or their owned children, held out of scope.

        Merge matches a row on its table's primary key, which may differ from the repository's
        id, and takes the scope's values where that key holds them.
        """
        if not self._reads.scope:
            return []

        mapper = sa_inspect(self._reads.content_type)
        names = [mapper.get_property_by_column(column).key for column in mapper.primary_key]
        keys = [
            list(starmap(self._reads.scope.get, zip(names, values, strict=True)))
            for values in (mapper.primary_key_from_instance(record) for record in records)
        ]
        return sorted(await self._outside(mapper, records, keys, self._reads.in_scope()))

    async def _outside(
        self,
        mapper: Mapper[Any],
        rows: Sequence[object],
        keys: Sequence[Sequence[Any]],
        in_scope: ColumnElement[bool],
    ) -> set[str]:
        """The keys of these rows held out of the scope, then of their owned children.

        Args:
            mapper: Maps the rows' table.
            rows: The rows named.
            keys: The primary key merge matches each row on.
            in_scope: The condition a row of this table meets when the scope holds it.

        Returns:
            The keys held out of the scope, as text.
        """
        outside = await self._held_keys(mapper, keys) - await self._held_keys(
            mapper, keys, in_scope
        )

        for relationship in self._owned(mapper):
            owned = relationship.mapper
            children = [child for row in rows for child in self._children(row, relationship)]

            if not children:
                continue

            # A child of the same table meets the scope's condition itself; another, through
            # its parent.
            child_in_scope = (
                in_scope if owned is mapper else exists().where(relationship.primaryjoin, in_scope)
            )
            child_keys = [owned.primary_key_from_instance(child) for child in children]
            outside |= await self._outside(owned, children, child_keys, child_in_scope)

        return outside

    async def _held_keys(
        self, mapper: Mapper[Any], keys: Sequence[Sequence[Any]], *conditions: ColumnElement[bool]
    ) -> set[str]:
        """The primary keys among these the table holds where the conditions hold, as text."""
        columns = list(mapper.primary_key)
        known = [key for key in keys if None not in key]
        held: set[str] = set()

        for batch in batched(known, self.KEYS_PER_STATEMENT // len(columns)):
            statement = select(*columns).where(self._keys_condition(columns, batch), *conditions)
            held |= {
                self._label(tuple(row)) for row in await self._reads.session.execute(statement)
            }

        return held

    @staticmethod
    def _keys_condition(
        columns: list[ColumnElement[Any]], keys: Sequence[Sequence[Any]]
    ) -> ColumnElement[bool]:
        """The condition that selects rows by their primary key, of one column or several."""
        if len(columns) == 1:
            return columns[0].in_([key[0] for key in keys])

        return or_(
            *(
                and_(*(column == value for column, value in zip(columns, key, strict=True)))
                for key in keys
            )
        )

    @staticmethod
    def _label(key: Sequence[Any]) -> str:
        """A primary key as text: its value, or its values in parentheses."""
        return str(key[0]) if len(key) == 1 else f"({', '.join(str(value) for value in key)})"

    @staticmethod
    def _owned(mapper: Mapper[Any]) -> list[RelationshipProperty[Any]]:
        """The relationships along which merge writes children this table owns."""
        return [
            relationship
            for relationship in mapper.relationships
            if relationship.direction is RelationshipDirection.ONETOMANY
            and relationship.cascade.merge
        ]

    @staticmethod
    def _children(record: object, relationship: RelationshipProperty[Any]) -> list[Any]:
        """The children a record gives along a relationship, none when it gives none."""
        children = sa_inspect(record).dict.get(relationship.key) or []
        return children if isinstance(children, list) else [children]

    def _leave_to_the_database(self, record: SQLModel) -> None:
        """Unset the empty columns the database fills, here and in owned children, so merge keeps them."""
        state = sa_inspect(record)

        for column in state.mapper.columns:
            if column.server_default is not None and state.dict.get(column.key) is None:
                state.dict.pop(column.key, None)

        for relationship in state.mapper.relationships:
            for child in self._children(record, relationship):
                self._leave_to_the_database(child)

    def _scoped(self, record: T) -> T:
        """The record with the scope's values set."""
        for name, value in self._reads.scope.items():
            setattr(record, name, value)

        return record

    def _id_of(self, record: T) -> str:
        """The record's id, as text."""
        return str(getattr(record, self._reads.primary_key))
