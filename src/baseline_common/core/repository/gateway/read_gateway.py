"""ReadGateway — reads one record type from an HTTP API, page after page, in a query dialect."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from baseline_common.core.repository.abstract_repository import AbstractReadRepository
from baseline_common.core.repository.gateway.abstract_query_dialect import (
    AbstractQueryDialect,
    UnreadableAnswerError,
)
from baseline_common.core.repository.gateway.http_client.http_client import HttpClient
from baseline_common.core.repository.query_filters import (
    EqualityFilter,
    Filter,
    LogicalFilter,
    SortOrder,
)


class UnmappedFieldError(ValueError):
    """A query names a field the model does not have."""


class ReadGateway[T: BaseModel](AbstractReadRepository[T]):
    """Reads one record type from an HTTP API; the model's aliases are the API's fields."""

    def __init__(
        self,
        client: HttpClient,
        content_type: type[T],
        dialect: AbstractQueryDialect,
        *,
        primary_key: str = "id",
    ) -> None:
        """Bind the client, the model and the dialect.

        Args:
            client: The client bound to the API.
            content_type: The model every record is read as; its aliases are the API's fields.
            dialect: How the API is queried.
            primary_key: The API field holding a record's primary key.
        """
        super().__init__(content_type)
        self._client = client
        self._dialect = dialect
        self._primary_key = primary_key

    async def get_by_id(self, id_: str) -> T | None:
        """Fetch a record by its id.

        Args:
            id_: The record's id in the API.

        Returns:
            The record, or None.
        """
        rows = await self._fetch([EqualityFilter(field=self._primary_key, value=id_)], limit=1)
        return rows[0] if rows else None

    async def get_by_filters(
        self,
        filters: Sequence[Filter] = (),
        sort_by: str | None = None,
        sort_order: SortOrder = SortOrder.ASC,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[T]:
        """Fetch the records every filter holds for.

        Args:
            filters: The conditions on the model's fields, all of which must hold.
            sort_by: The model field to sort on.
            sort_order: The order to sort in.
            limit: The most records returned; None fetches them all.
            offset: How many matching records to skip.

        Returns:
            The matching records.
        """
        return await self._fetch(
            [self._on_api_fields(condition) for condition in filters],
            self._api_name(sort_by) if sort_by else None,
            sort_order,
            limit,
            offset,
        )

    async def _fetch(
        self,
        filters: Sequence[Filter],
        sort_by: str | None = None,
        sort_order: SortOrder = SortOrder.ASC,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[T]:
        """Fetch pages until the limit or the last record.

        Args:
            filters: The conditions, on the API's fields.
            sort_by: The API field to sort on.
            sort_order: The order to sort in.
            limit: The most records returned; None fetches them all.
            offset: How many matching records to skip.

        Returns:
            The records.

        Raises:
            UnreadableAnswerError: When a record the API answers does not fit the model.
        """
        fields = [self._api_name(name) for name in self.content_type.model_fields]
        rows: list[dict[str, Any]] = []

        while limit is None or len(rows) < limit:
            size = self._dialect.page_size
            size = size if limit is None else min(size, limit - len(rows))
            request = self._dialect.page_request(
                fields, filters, sort_by, sort_order, offset + len(rows), size
            )
            response = await self._client.request(
                request.method, request.path, params=request.params, json=request.json_body
            )
            page, more = self._dialect.read_page(response)
            rows.extend(page)

            if not more:
                break

        # A dialect that cannot page answers every row: the limit is kept here all the same.
        if limit is not None:
            rows = rows[:limit]

        try:
            return [self.content_type.model_validate(row) for row in rows]
        except ValidationError as error:
            detail = error.errors()[0]
            field = ".".join(map(str, detail["loc"]))
            msg = (
                f"{self._dialect.source}: unreadable row, {field}: {detail['msg']} "
                f"({str(detail.get('input'))[:120]})"
            )
            raise UnreadableAnswerError(msg) from error

    def _on_api_fields(self, condition: Filter) -> Filter:
        """Rename a filter's fields to the API's.

        Args:
            condition: A filter on the model's fields.

        Returns:
            The same filter on the API's fields.
        """
        if isinstance(condition, LogicalFilter):
            operands = [self._on_api_fields(operand) for operand in condition.operands]
            return condition.model_copy(update={"operands": operands})

        field = getattr(condition, "field", None)

        if field is None:
            return condition

        return condition.model_copy(update={"field": self._api_name(field)})

    def _api_name(self, name: str) -> str:
        """Find the API field a model field reads: its alias, else its name.

        Args:
            name: The model field.

        Returns:
            The API field's name.

        Raises:
            UnmappedFieldError: When the model has no such field.
        """
        field = self.content_type.model_fields.get(name)

        if field is None:
            msg = f"{self.content_type.__name__} has no field {name}"
            raise UnmappedFieldError(msg)

        return field.alias or name
