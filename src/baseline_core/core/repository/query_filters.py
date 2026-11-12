"""The filters a repository query is made of: compiled by a backend, or matched in Python."""

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import date
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict


class SortOrder(StrEnum):
    """The order a query's results are sorted in."""

    ASC = "asc"
    DESC = "desc"


class Filter(BaseModel, ABC):
    """A condition on the records a query returns, on a field named by its model path."""

    model_config = ConfigDict(frozen=True)

    operator: Literal["==", "!=", ">", "<", ">=", "<=", "&", "|", "in"]

    @abstractmethod
    def matches(self, record: BaseModel) -> bool:
        """Tell whether the condition holds for a record.

        Args:
            record: The record.

        Returns:
            Whether it holds.
        """

    @staticmethod
    def value_of(record: BaseModel, path: str) -> Any:  # noqa: ANN401
        """Read a field by its dotted path, None when a step on the way is missing.

        Args:
            record: The record.
            path: The field path, e.g. ``civic_address.city``.

        Returns:
            The field's value.
        """
        value: Any = record

        for name in path.split("."):
            value = getattr(value, name, None)

        return value

    @staticmethod
    def folded(value: Any) -> Any:  # noqa: ANN401
        """Case-fold text so it compares regardless of case, and write a date as a filter does.

        Args:
            value: A field's value, or a filter's.

        Returns:
            The text case-folded, a date in ISO form, any other value unchanged.
        """
        if isinstance(value, date):
            return value.isoformat()

        return value.casefold() if isinstance(value, str) else value


class LogicalFilter(Filter):
    """Every operand holds (``&``), or any one of them does (``|``)."""

    operator: Literal["&", "|"]
    operands: Sequence[Filter]

    def matches(self, record: BaseModel) -> bool:
        """Tell whether every operand, or any one of them, holds for a record.

        Args:
            record: The record.

        Returns:
            Whether it holds.
        """
        held = (operand.matches(record) for operand in self.operands)
        return all(held) if self.operator == "&" else any(held)


class ComparisonFilter(Filter):
    """A field compared with a value; text compared regardless of case."""

    COMPARISONS: ClassVar[dict[str, Callable[[Any, Any], bool]]] = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
    }

    operator: Literal["==", "!=", ">", "<", ">=", "<="]
    field: str
    value: str | float | int | bool

    def matches(self, record: BaseModel) -> bool:
        """Compare the record's field with the value; a missing field only differs.

        Args:
            record: The record.

        Returns:
            Whether it holds.
        """
        value = self.value_of(record, self.field)

        if value is None:
            return self.operator == "!="

        compare = self.COMPARISONS[self.operator]
        return compare(self.folded(value), self.folded(self.value))


class EqualityFilter(ComparisonFilter):
    """A field equal to a value."""

    operator: Literal["=="] = "=="


class TextFilter(Filter):
    """A text field containing a value, regardless of case."""

    operator: Literal["in"] = "in"
    field: str
    value: str

    def matches(self, record: BaseModel) -> bool:
        """Tell whether the record's text field contains the value.

        Args:
            record: The record.

        Returns:
            Whether it holds.
        """
        value = self.value_of(record, self.field)
        return isinstance(value, str) and self.folded(self.value) in self.folded(value)
