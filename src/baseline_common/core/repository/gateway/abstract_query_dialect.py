"""AbstractQueryDialect — how an API is queried: a page's request, and how its answer reads."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from http import HTTPMethod
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from baseline_common.core.repository.query_filters import Filter, SortOrder


class UnreadableAnswerError(RuntimeError):
    """An API answered with what cannot be read as its records."""


class PageRequest(BaseModel):
    """The HTTP request fetching one page."""

    model_config = ConfigDict(frozen=True)

    method: HTTPMethod
    path: str
    params: dict[str, Any] | None = None
    json_body: dict[str, Any] | None = None


class AbstractQueryDialect(ABC):
    """How an API is queried, every field named as the API names it."""

    def __init__(self, page_size: int) -> None:
        """Bind the page size.

        Args:
            page_size: Records per page, at most.

        Raises:
            ValueError: When the page size is not positive.
        """
        if page_size <= 0:
            msg = f"page_size must be positive, not {page_size}"
            raise ValueError(msg)

        self.page_size = page_size

    @property
    def source(self) -> str:
        """What the records are read from, as a person names it."""
        return type(self).__name__

    @abstractmethod
    def page_request(
        self,
        fields: list[str],
        filters: Sequence[Filter],
        sort_by: str | None,
        sort_order: SortOrder,
        offset: int,
        size: int,
    ) -> PageRequest:
        """Build the request fetching one page.

        Args:
            fields: The API fields to fetch.
            filters: The conditions on API fields, all of which must hold.
            sort_by: The API field to sort on.
            sort_order: The order to sort in.
            offset: How many matching records to skip.
            size: Records in the page, at most.

        Returns:
            The request.
        """

    @abstractmethod
    def read_page(self, response: httpx.Response) -> tuple[list[dict[str, Any]], bool]:
        """Read a page.

        Args:
            response: The API's answer.

        Returns:
            The records, and whether more follow.
        """
