"""PostgresReadRepository — reads one table of SQLModel records, through the caller's session."""

from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import ColumnElement, String, TypeDecorator, and_, func, or_, true
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.types import TypeEngine
from sqlmodel import SQLModel, select

from baseline_common.core.repository.abstract_repository import AbstractReadRepository
from baseline_common.core.repository.query_filters import (
    ComparisonFilter,
    Filter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)


class PostgresReadRepository[T: SQLModel](AbstractReadRepository[T]):
    """Reads the rows of one table, within a fixed scope such as one term."""

    def __init__(
        self,
        session: AsyncSession,
        content_type: type[T],
        id_field: str = "id",
        scope: dict[str, Any] | None = None,
    ) -> None:
        """Bind the session, the table, the column that identifies a row, and the scope.

        Args:
            session: The caller's session; the caller commits it.
            content_type: The SQLModel table class.
            id_field: The column a row's id is read from.
            scope: Column values every row read or written has, e.g. ``{"term_id": term_id}``.
        """
        super().__init__(content_type)
        self.session = session
        self.id_field = id_field
        self.scope = scope or {}

    async def get_by_id(self, id_: str) -> T | None:
        """Find a row of the scope by its id.

        Args:
            id_: The row's id.

        Returns:
            The row, or None when there is none.
        """
        statement = select(self.content_type).where(self.key_condition([id_]), self.in_scope())
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def get_by_filters(
        self,
        filters: Sequence[Filter] = (),
        sort_by: str | None = None,
        sort_order: SortOrder = SortOrder.ASC,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[T]:
        """Find the rows of the scope every filter holds for.

        Args:
            filters: The conditions, all of which must hold; none returns every row.
            sort_by: The column to sort on.
            sort_order: The order to sort in.
            limit: The most rows returned; None returns them all.
            offset: How many matching rows to skip.

        Returns:
            The matching rows.
        """
        statement = select(self.content_type).where(
            self.in_scope(), *(self.condition(filter_) for filter_ in filters)
        )

        if sort_by is not None:
            column = self.column(sort_by)
            statement = statement.order_by(
                column.desc() if sort_order is SortOrder.DESC else column
            )

        statement = statement.offset(offset).limit(limit)
        return list((await self.session.execute(statement)).scalars().all())

    def column(self, name: str) -> InstrumentedAttribute[Any]:
        """The table's column attribute.

        Args:
            name: The column's name.

        Returns:
            The column attribute.
        """
        attribute: InstrumentedAttribute[Any] = getattr(self.content_type, name)
        return attribute

    def coerced(self, name: str, value: Any) -> Any:  # noqa: ANN401
        """A value as the column stores it, e.g. a text id as a UUID.

        Args:
            name: The column's name.
            value: The value.

        Returns:
            The value in the column's Python type.
        """
        python_type = self.sql_type(name).python_type
        return value if isinstance(value, python_type) else python_type(value)

    def sql_type(self, name: str) -> TypeEngine[Any]:
        """The column's SQL type, under any decorator such as SQLModel's ``AutoString``.

        Args:
            name: The column's name.

        Returns:
            The type.
        """
        type_ = self.column(name).type
        return type_.impl_instance if isinstance(type_, TypeDecorator) else type_

    def key_condition(self, ids: Sequence[str]) -> ColumnElement[bool]:
        """The condition that selects rows by id.

        Args:
            ids: The rows' ids.

        Returns:
            The condition.
        """
        return self.column(self.id_field).in_([self.coerced(self.id_field, id_) for id_ in ids])

    def in_scope(self) -> ColumnElement[bool]:
        """The condition every row of the scope meets.

        Returns:
            The condition; always true without a scope.
        """
        return and_(true(), *(self.column(name) == value for name, value in self.scope.items()))

    def condition(self, filter_: Filter) -> ColumnElement[bool]:
        """Compile a filter to SQL, comparing text regardless of case as the contract says.

        Args:
            filter_: The filter.

        Returns:
            The SQL condition.

        Raises:
            NotImplementedError: For a filter type no backend query needs yet.
        """
        if isinstance(filter_, LogicalFilter):
            operands = [self.condition(operand) for operand in filter_.operands]
            return and_(*operands) if filter_.operator == "&" else or_(*operands)

        if isinstance(filter_, TextFilter):
            return self.column(filter_.field).ilike(f"%{filter_.value}%")

        if isinstance(filter_, ComparisonFilter):
            left: InstrumentedAttribute[Any] | ColumnElement[Any] = self.column(filter_.field)
            value = filter_.value

            if isinstance(value, str) and isinstance(self.sql_type(filter_.field), String):
                left, value = func.lower(left), value.casefold()
            else:
                value = self.coerced(filter_.field, value)

            # The comparison functions return SQL expressions when given a column.
            compare = ComparisonFilter.COMPARISONS[filter_.operator]
            return cast("ColumnElement[bool]", compare(left, value))

        raise NotImplementedError(type(filter_).__name__)
