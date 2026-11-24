"""Unit tests for the read gateway over a Fidelio analysis view: GetView, the filters as SQL."""

import json
from collections.abc import Callable
from datetime import date
from decimal import Decimal

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from baseline_common.framework.repository.http.abstract_query_dialect import UnreadableAnswerError
from baseline_common.framework.repository.http.client.http_client import HttpClient
from baseline_common.framework.repository.http.fidelio.view_dialect import FidelioViewDialect
from baseline_common.framework.repository.http.read_gateway import ReadGateway
from baseline_common.framework.repository.query_filters import (
    ComparisonFilter,
    EqualityFilter,
    LogicalFilter,
    TextFilter,
)


class InvoicedTotal(BaseModel):
    """What a view sums as invoiced to one client on one date; each alias is a column."""

    model_config = ConfigDict(frozen=True, validate_by_name=True, validate_by_alias=True)
    """Frozen, and built by field name as well as by column."""

    invoiced_on: date = Field(alias="date")
    """The date invoiced, column ``date``."""
    client_number: str = Field(alias="code_client", min_length=1)
    """The client's number, column ``code_client``."""
    quantity: Decimal = Field(alias="quantite", allow_inf_nan=False)
    """The quantity invoiced, column ``quantite``."""
    net_amount: Decimal = Field(alias="montant_net", allow_inf_nan=False)
    """The net amount invoiced, column ``montant_net``."""


# How the scripted Fidelio server answers; `ViewOf` and the gateway it builds take one.
type Handler = Callable[[httpx.Request], httpx.Response]

# The signature of the `view_of` fixture, which every test of the file annotates.
type ViewOf = Callable[[Handler], ReadGateway[InvoicedTotal]]


class TestView:
    """Filters on the model's fields become one SQL condition on the view's columns."""

    @pytest.fixture
    def view_of(self) -> ViewOf:
        """Return a builder of gateways over the invoiced view, Fidelio played by a handler."""

        def build(handler: Handler) -> ReadGateway[InvoicedTotal]:
            client = HttpClient(
                "https://fidelio.test/Heimdall/PhoenixService.svc/Rest",
                transport=httpx.MockTransport(handler),
            )
            dialect = FidelioViewDialect("ADV_Ventes_Clients", "env-123")
            return ReadGateway(client, InvoicedTotal, dialect, primary_key="client_number")

        return build

    @staticmethod
    def row(client_number: str = "CLI0000187") -> dict[str, object]:
        """Return one row of the view, as Fidelio answers it.

        Args:
            client_number: The client's number.

        Returns:
            The row.
        """
        return {
            "date": "2026-04-01T00:00:00-04:00",
            "code_client": client_number,
            "quantite": 1250.0,
            "montant": 98765.4,
            "montant_net": 97000.25,
        }

    @staticmethod
    def sent_body(request: httpx.Request) -> dict[str, object]:
        """Read the JSON a request carries.

        Args:
            request: A request to the view.

        Returns:
            The body.
        """
        body: dict[str, object] = json.loads(request.content)
        return body

    async def test__when_two_filters_are_given__become_one_condition_on_the_columns(
        self, view_of: ViewOf
    ) -> None:
        """Two conditions: an ``AND``, each naming the column; the environment and view beside."""
        sent: list[httpx.Request] = []

        def answer(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={"changes": [self.row()]})

        rows = await view_of(answer).get_by_filters([
            EqualityFilter(field="client_number", value="CLI0000187"),
            ComparisonFilter(field="invoiced_on", operator=">=", value="2026-04-01"),
        ])

        (request,) = sent
        assert (request.method, request.url.path) == (
            "POST",
            "/Heimdall/PhoenixService.svc/Rest/GetView",
        )
        assert self.sent_body(request) == {
            "environmentName": "env-123",
            "viewName": "ADV_Ventes_Clients",
            "filter": "(code_client = 'CLI0000187' AND date >= '2026-04-01')",
        }
        (invoiced,) = rows
        assert (invoiced.client_number, invoiced.invoiced_on, invoiced.net_amount) == (
            "CLI0000187",
            date(2026, 4, 1),
            Decimal("97000.25"),
        )

    async def test__when_one_condition_is_given__is_bare_and_a_text_search_a_like(
        self, view_of: ViewOf
    ) -> None:
        """A single term needs no brackets; an ``OR`` keeps its own; a quote is doubled."""
        sent: list[httpx.Request] = []

        def answer(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={"changes": []})

        await view_of(answer).get_by_filters([
            LogicalFilter(
                operator="|",
                operands=[
                    TextFilter(field="client_number", value="CLI"),
                    ComparisonFilter(field="client_number", operator="!=", value="O'Brien"),
                ],
            )
        ])

        assert self.sent_body(sent[0])["filter"] == (
            "(code_client LIKE '%CLI%' OR code_client <> 'O''Brien')"
        )

    async def test__when_no_filter_is_given__sends_no_condition(self, view_of: ViewOf) -> None:
        """The whole view: the environment and its name alone."""
        sent: list[httpx.Request] = []

        def answer(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={"changes": []})

        await view_of(answer).get_by_filters()

        assert self.sent_body(sent[0]) == {
            "environmentName": "env-123",
            "viewName": "ADV_Ventes_Clients",
        }

    async def test__when_a_limit_is_given__every_row_comes_in_one_answer_and_the_limit_is_kept(
        self, view_of: ViewOf
    ) -> None:
        """Three rows in one answer: one request, the first two kept for a limit of 2."""
        sent: list[httpx.Request] = []

        def answer(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(
                200, json={"changes": [self.row("A"), self.row("B"), self.row("C")]}
            )

        rows = await view_of(answer).get_by_filters(limit=2)

        assert [row.client_number for row in rows] == ["A", "B"]
        assert len(sent) == 1

    async def test__when_a_sort_or_an_offset_is_asked__is_refused(self, view_of: ViewOf) -> None:
        """The view neither sorts nor skips: asking is a programming error, not a quiet misread."""
        view = view_of(lambda _: httpx.Response(200, json={"changes": []}))

        with pytest.raises(ValueError, match="neither sorts nor skips"):
            await view.get_by_filters(sort_by="invoiced_on")

        with pytest.raises(ValueError, match="neither sorts nor skips"):
            await view.get_by_filters(offset=1)

    async def test__when_an_answer_is_empty__is_no_row(self, view_of: ViewOf) -> None:
        """A view answering nothing at all is read as no row, not as an error."""
        rows = await view_of(lambda _: httpx.Response(200)).get_by_filters()

        assert rows == []

    async def test__when_a_refusal_comes_in_text__is_raised_with_the_view_and_fidelio_s_words(
        self, view_of: ViewOf
    ) -> None:
        """Fidelio refuses a bad credential or an unknown view with a 200 and a sentence."""
        view = view_of(lambda _: httpx.Response(200, text="You don't have access to this view"))

        with pytest.raises(UnreadableAnswerError, match="view ADV_Ventes_Clients: You"):
            await view.get_by_filters()

    async def test__when_a_row_does_not_fit__names_the_view_and_the_field(
        self, view_of: ViewOf
    ) -> None:
        """A row without its client is Fidelio's fault, not the caller's: the field is named."""
        row = {**self.row(), "code_client": None}
        view = view_of(lambda _: httpx.Response(200, json={"changes": [row]}))

        with pytest.raises(
            UnreadableAnswerError,
            match="Fidelio view ADV_Ventes_Clients: unreadable row, code_client",
        ):
            await view.get_by_filters()
