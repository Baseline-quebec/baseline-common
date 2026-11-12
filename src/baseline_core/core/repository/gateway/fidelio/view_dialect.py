"""FidelioViewDialect — reads one analysis view of Fidelio with GetView, the filters as SQL."""

from collections.abc import Sequence
from http import HTTPMethod
from typing import Any, ClassVar, override

import httpx

from baseline_core.core.repository.gateway.abstract_query_dialect import (
    AbstractQueryDialect,
    PageRequest,
    UnreadableAnswerError,
)
from baseline_core.core.repository.query_filters import (
    ComparisonFilter,
    Filter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)


class FidelioViewDialect(AbstractQueryDialect):
    """Reads one analysis view with ``GetView``: every matching row at once, no sort, no offset."""

    PATH: ClassVar[str] = "GetView"
    ROWS: ClassVar[str] = "changes"
    """The key the rows come under."""

    ALL_ROWS: ClassVar[int] = 1_000_000
    """A nominal page size: the view answers every row, whatever is asked."""

    COMPARISONS: ClassVar[dict[str, str]] = {"==": "=", "!=": "<>"}
    """The operators SQL writes otherwise."""

    def __init__(self, view: str, environment: str) -> None:
        """Bind the view and the environment.

        Args:
            view: The view's name, ``ADV_Contrats``.
            environment: The environment name, Fidelio's credential.
        """
        super().__init__(self.ALL_ROWS)
        self._view = view
        self._environment = environment

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
        """Build the one request reading the view.

        Args:
            fields: The columns; the view's are fixed, so they are not sent.
            filters: The conditions on the view's columns, all of which must hold.
            sort_by: Refused: the view does not sort.
            sort_order: The order to sort in.
            offset: Refused past zero: the view does not skip.
            size: Ignored: the view answers every row.

        Returns:
            The request.

        Raises:
            ValueError: When asked to sort or to skip rows.
        """
        if sort_by or offset:
            msg = f"a Fidelio view neither sorts nor skips rows ({self._view})"
            raise ValueError(msg)

        body: dict[str, Any] = {"environmentName": self._environment, "viewName": self._view}

        if (condition := self._compile(LogicalFilter(operator="&", operands=filters))) is not None:
            body["filter"] = condition

        return PageRequest(method=HTTPMethod.POST, path=self.PATH, json_body=body)

    @override
    def read_page(self, response: httpx.Response) -> tuple[list[dict[str, Any]], bool]:
        """Read every row of the view, under ``changes``.

        Args:
            response: Fidelio's answer.

        Returns:
            The rows, and False: nothing follows.

        Raises:
            UnreadableAnswerError: When Fidelio answers with text, as it refuses a view.
        """
        if not response.content:
            return [], False

        try:
            body = response.json()
        except ValueError:
            body = None

        if not isinstance(body, dict) or self.ROWS not in body:
            msg = f"Fidelio refused the view {self._view}: {response.text.strip()[:200]}"
            raise UnreadableAnswerError(msg)

        rows: list[dict[str, Any]] = body[self.ROWS]
        return rows, False

    @property
    @override
    def source(self) -> str:
        """The view, as a person names it."""
        return f"Fidelio view {self._view}"

    @classmethod
    def _compile(cls, condition: Filter) -> str | None:
        """Compile a filter into one SQL condition on the view's columns.

        Args:
            condition: The filter, on the view's columns.

        Returns:
            The condition; None for a logical filter without operands.

        Raises:
            TypeError: For a filter type the view cannot express.
        """
        if isinstance(condition, LogicalFilter):
            terms = [cls._compile(operand) for operand in condition.operands]
            kept = [term for term in terms if term is not None]

            if not kept:
                return None

            if len(kept) == 1:
                return kept[0]

            return f"({(' AND ' if condition.operator == '&' else ' OR ').join(kept)})"

        if isinstance(condition, TextFilter):
            return f"{condition.field} LIKE {cls._literal(f'%{condition.value}%')}"

        if isinstance(condition, ComparisonFilter):
            operator = cls.COMPARISONS.get(condition.operator, condition.operator)
            return f"{condition.field} {operator} {cls._literal(condition.value)}"

        msg = f"a Fidelio view cannot express a {type(condition).__name__}"
        raise TypeError(msg)

    @staticmethod
    def _literal(value: str | float | bool) -> str:  # noqa: FBT001 - a value, not a flag
        """Write a value as SQL does: text quoted, its quotes doubled; a bit as 1 or 0.

        Args:
            value: The filter's value.

        Returns:
            The literal.
        """
        if isinstance(value, bool):
            return "1" if value else "0"

        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"

        return str(value)
