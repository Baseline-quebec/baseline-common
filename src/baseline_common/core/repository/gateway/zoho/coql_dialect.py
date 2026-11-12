"""ZohoCoqlDialect — queries a Zoho CRM module with COQL."""

from collections.abc import Sequence
from functools import reduce
from http import HTTPMethod
from typing import Any, ClassVar, override

import httpx

from baseline_common.core.repository.gateway.abstract_query_dialect import (
    AbstractQueryDialect,
    PageRequest,
)
from baseline_common.core.repository.query_filters import (
    ComparisonFilter,
    Filter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)


class ZohoCoqlDialect(AbstractQueryDialect):
    """Queries a Zoho CRM module with COQL, every pair of terms in brackets."""

    QUERY_PATH: ClassVar[str] = "coql"
    MAX_PAGE_SIZE: ClassVar[int] = 2000
    COMPARISONS: ClassVar[dict[str, str]] = {"==": "="}

    def __init__(self, module: str, page_size: int = MAX_PAGE_SIZE) -> None:
        """Bind the module.

        Args:
            module: The module's API name, e.g. ``Accounts``.
            page_size: Records per query, at most.
        """
        super().__init__(page_size)
        self._module = module

    @property
    @override
    def source(self) -> str:
        """The module, as a person names it."""
        return f"Zoho module {self._module}"

    @override
    def page_request(
        self,
        fields: list[str],
        filters: Sequence[Filter],
        sort_by: str | None,
        sort_order: SortOrder,
        offset: int,
        size: int,
    ) -> PageRequest:
        """Build the COQL query of one page.

        Args:
            fields: The Zoho fields to select.
            filters: The conditions on Zoho fields, all of which must hold.
            sort_by: The Zoho field to sort on.
            sort_order: The order to sort in.
            offset: How many matching records to skip.
            size: Records in the page, at most.

        Returns:
            The request.
        """
        where = self._compile(LogicalFilter(operator="&", operands=filters)) or "id is not null"
        order = f" order by {sort_by} {sort_order}" if sort_by else ""
        query = (
            f"select {', '.join(fields)} from {self._module} "  # noqa: S608 - literals escaped
            f"where {where}{order} limit {offset}, {size}"
        )
        return PageRequest(
            method=HTTPMethod.POST, path=self.QUERY_PATH, json_body={"select_query": query}
        )

    @override
    def read_page(self, response: httpx.Response) -> tuple[list[dict[str, Any]], bool]:
        """Read a page; Zoho answers an empty 204 when nothing matches.

        Args:
            response: Zoho's answer.

        Returns:
            The records, and whether more follow.
        """
        if not response.content:
            return [], False

        body = response.json()
        return body.get("data", []), bool(body.get("info", {}).get("more_records"))

    def _compile(self, condition: Filter) -> str:
        """Compile a filter into a COQL condition.

        Args:
            condition: The filter, on Zoho fields.

        Returns:
            The condition; empty for a logical filter without operands.

        Raises:
            TypeError: For a filter type COQL cannot express.
        """
        if isinstance(condition, LogicalFilter):
            joiner = " and " if condition.operator == "&" else " or "
            terms = [self._compile(operand) for operand in condition.operands]
            return reduce(lambda left, right: f"({left}{joiner}{right})", terms) if terms else ""

        if isinstance(condition, TextFilter):
            return f"{condition.field} like {self._literal(f'%{condition.value}%')}"

        if isinstance(condition, ComparisonFilter):
            comparison = self.COMPARISONS.get(condition.operator, condition.operator)
            return f"{condition.field} {comparison} {self._literal(condition.value)}"

        msg = f"COQL cannot express a {type(condition).__name__}"
        raise TypeError(msg)

    @staticmethod
    def _literal(value: str | float | bool) -> str:  # noqa: FBT001 - a filter value, not a flag
        """Render a value as a COQL literal.

        Args:
            value: The value.

        Returns:
            The literal: quotes doubled, a boolean, or a number bracketed when negative.
        """
        if isinstance(value, bool):
            return str(value).lower()

        if isinstance(value, int | float):
            return f"({value})" if value < 0 else str(value)

        escaped = value.replace("'", "''")
        return f"'{escaped}'"
