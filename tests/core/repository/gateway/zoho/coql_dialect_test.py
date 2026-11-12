"""Unit tests for the Zoho COQL dialect: filters as COQL queries, pages read."""

import json

import httpx
import pytest

from baseline_common.core.repository.gateway.read_gateway import UnmappedFieldError
from baseline_common.core.repository.gateway.zoho.coql_dialect import ZohoCoqlDialect
from baseline_common.core.repository.query_filters import (
    ComparisonFilter,
    EqualityFilter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)
from tests.core.repository.gateway.http_client.conftest import ScriptedServer
from tests.core.repository.gateway.zoho.conftest import ReadsOf, ZohoAccount


class TestQuery:
    """Filters on the model's fields become a COQL query on their aliases."""

    @staticmethod
    def sent_query(request: httpx.Request) -> str:
        """Read the COQL query a request carries.

        Args:
            request: A request to the CRM.

        Returns:
            The query.
        """
        return str(json.loads(request.content)["select_query"])

    async def test__filters__are_compiled_in_bracketed_pairs_on_zoho_fields(
        self, reads_of: ReadsOf
    ) -> None:
        """Three conditions: COQL wants every pair of terms in brackets."""
        crm = ScriptedServer(httpx.Response(204))
        reads = reads_of(crm)
        filters = [
            EqualityFilter(field="status", value="Actif"),
            LogicalFilter(
                operator="|",
                operands=[
                    EqualityFilter(field="industry", value="Foundry"),
                    TextFilter(field="postal_code", value="G1K"),
                ],
            ),
            EqualityFilter(field="name", value="ATELIER D'ANJOU"),
        ]

        await reads.get_by_filters(filters, sort_by="name", sort_order=SortOrder.DESC, limit=50)

        assert crm.paths() == ["/crm/v8/coql"]
        assert self.sent_query(crm.requests[0]) == (
            "select id, Account_Name, Industry, Status, Billing_Code from Accounts "
            "where ((Status = 'Actif' and (Industry = 'Foundry' or Billing_Code like '%G1K%')) "
            "and Account_Name = 'ATELIER D''ANJOU') "
            "order by Account_Name desc limit 0, 50"
        )

    @pytest.mark.parametrize(
        ("value", "literal"),
        [(-53.214831, "(-53.214831)"), (-3, "(-3)"), (47.694201, "47.694201"), (0, "0")],
    )
    def test__a_negative_number__is_bracketed(self, value: float, literal: str) -> None:
        """COQL refuses a bare negative, e.g. a Canadian longitude."""
        dialect = ZohoCoqlDialect("Accounts")
        bound = ComparisonFilter(operator=">=", field="Longitude", value=value)

        request = dialect.page_request(["id"], [bound], None, SortOrder.ASC, 0, 1)

        assert request.json_body is not None
        assert request.json_body["select_query"].endswith(f"Longitude >= {literal} limit 0, 1")

    async def test__no_filter__still_gives_a_where_clause(self, reads_of: ReadsOf) -> None:
        """Every record of the module."""
        crm = ScriptedServer(httpx.Response(204))
        reads = reads_of(crm)

        found = await reads.get_by_filters()

        assert found == []
        assert "where id is not null limit 0, 100" in self.sent_query(crm.requests[0])

    async def test__a_field_the_model_lacks__is_refused_naming_it(self, reads_of: ReadsOf) -> None:
        """Nothing is sent."""
        crm = ScriptedServer(httpx.Response(204))
        reads = reads_of(crm)

        with pytest.raises(UnmappedFieldError, match="owner_name"):
            await reads.get_by_filters([EqualityFilter(field="owner_name", value="X")])

        assert crm.requests == []


class TestPaging:
    """Pages follow one another until Zoho says no more records follow."""

    async def test__every_page__is_fetched_without_a_limit(self, reads_of: ReadsOf) -> None:
        """The second page starts after the records already read."""
        first = {"id": "z-1", "Account_Name": "A", "Status": "Actif"}
        second = {"id": "z-2", "Account_Name": "B", "Status": "Actif"}
        crm = ScriptedServer(
            httpx.Response(200, json={"data": [first], "info": {"more_records": True}}),
            httpx.Response(200, json={"data": [second], "info": {"more_records": False}}),
        )
        reads = reads_of(crm)

        found = await reads.get_by_filters(limit=None)

        assert [account.zoho_id for account in found] == ["z-1", "z-2"]
        assert TestQuery.sent_query(crm.requests[0]).endswith("limit 0, 2000")
        assert TestQuery.sent_query(crm.requests[1]).endswith("limit 1, 2000")


class TestGetById:
    """A record read back as the model, from its aliases."""

    async def test__a_record__is_read_as_the_model(
        self, reads_of: ReadsOf, account: ZohoAccount
    ) -> None:
        """Zoho's fields become the model's."""
        record = account.model_dump(by_alias=True)
        crm = ScriptedServer(httpx.Response(200, json={"data": [record], "info": {}}))
        reads = reads_of(crm)

        found = await reads.get_by_id("z-1")

        assert "where id = 'z-1' limit 0, 1" in TestQuery.sent_query(crm.requests[0])
        assert found == account

    async def test__no_record__is_none(self, reads_of: ReadsOf) -> None:
        """Zoho answers an empty 204."""
        reads = reads_of(ScriptedServer(httpx.Response(204)))

        assert await reads.get_by_id("z-9") is None
