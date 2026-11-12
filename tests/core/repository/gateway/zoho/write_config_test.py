"""Unit tests for the Zoho write config: records written by their aliases, through the API."""

import json

import httpx
import pytest

from baseline_core.core.repository.gateway.write_gateway import WriteError
from tests.core.repository.gateway.http_client.conftest import ScriptedServer
from tests.core.repository.gateway.zoho.conftest import WritesOf, ZohoAccount


class TestWrites:
    """Records written by their aliases, the id left out on insert."""

    @staticmethod
    def sent_rows(request: httpx.Request) -> list[dict[str, object]]:
        """Read the records a write request carries.

        Args:
            request: A request to the CRM.

        Returns:
            The records, in Zoho fields.
        """
        rows: list[dict[str, object]] = json.loads(request.content)["data"]
        return rows

    @staticmethod
    def outcome(*ids: str) -> httpx.Response:
        """Answer a write as Zoho does when every record succeeds.

        Args:
            *ids: The records' ids.

        Returns:
            The answer.
        """
        data = [{"code": "SUCCESS", "status": "success", "details": {"id": i}} for i in ids]
        return httpx.Response(201, json={"data": data})

    async def test__an_insert__posts_the_record_without_its_id(
        self, writes_of: WritesOf, account: ZohoAccount
    ) -> None:
        """Zoho gives the id."""
        crm = ScriptedServer(self.outcome("z-10"))
        writes = writes_of(crm)

        id_ = await writes.insert_one(account)

        assert id_ == "z-10"
        assert (crm.requests[0].method, crm.paths()) == ("POST", ["/crm/v8/Accounts"])
        assert self.sent_rows(crm.requests[0]) == [
            {
                "Account_Name": "ATELIER TREMBLAY ET GAGNON",
                "Industry": "Machining",
                "Status": "Actif",
                "Billing_Code": "G1K 3X9",
            }
        ]

    async def test__an_update__puts_the_record_with_its_id(
        self, writes_of: WritesOf, account: ZohoAccount
    ) -> None:
        """Zoho finds the record to update by its id."""
        crm = ScriptedServer(self.outcome("z-1"))
        writes = writes_of(crm)
        closed = account.model_copy(update={"status": "Fermé"})

        await writes.update_one(closed)

        (row,) = self.sent_rows(crm.requests[0])
        assert crm.requests[0].method == "PUT"
        assert (row["id"], row["Status"]) == ("z-1", "Fermé")

    async def test__an_upsert__posts_to_the_upsert_path_with_the_ids_it_has(
        self, writes_of: WritesOf, account: ZohoAccount
    ) -> None:
        """A record with an id is updated, one without is inserted: Zoho tells them apart."""
        crm = ScriptedServer(self.outcome("z-1", "z-11"))
        writes = writes_of(crm)
        new = account.model_copy(update={"zoho_id": None, "name": "ATELIER NOUVEAU"})

        ids = await writes.upsert_many([account, new])

        assert ids == ["z-1", "z-11"]
        assert (crm.requests[0].method, crm.paths()) == ("POST", ["/crm/v8/Accounts/upsert"])
        held, added = self.sent_rows(crm.requests[0])
        assert held["id"] == "z-1"
        assert "id" not in added

    async def test__an_upsert__sends_the_fields_it_matches_on(
        self, writes_of: WritesOf, account: ZohoAccount
    ) -> None:
        """Zoho matches an upsert on the fields named beside the records."""
        crm = ScriptedServer(self.outcome("z-1"))
        writes = writes_of(crm, upsert_body={"duplicate_check_fields": ["Account_Name"]})

        await writes.upsert_one(account)

        body = json.loads(crm.requests[0].content)
        assert body["duplicate_check_fields"] == ["Account_Name"]
        assert len(body["data"]) == 1

    async def test__a_refused_record__raises_with_zohos_reason(
        self, writes_of: WritesOf, account: ZohoAccount
    ) -> None:
        """Zoho answers 400 with the reason per record."""
        refusal = {"code": "MANDATORY_NOT_FOUND", "status": "error", "message": "required field"}
        writes = writes_of(ScriptedServer(httpx.Response(400, json={"data": [refusal]})))

        with pytest.raises(WriteError, match="MANDATORY_NOT_FOUND"):
            await writes.insert_one(account)

    async def test__a_delete__names_the_record_in_the_ids_parameter(
        self, writes_of: WritesOf
    ) -> None:
        """One record, by its id, on the collection's path."""
        crm = ScriptedServer(self.outcome("z-1"))
        writes = writes_of(crm)

        id_ = await writes.delete_one("z-1")

        assert id_ == "z-1"
        assert (crm.requests[0].method, crm.paths()) == ("DELETE", ["/crm/v8/Accounts"])
        assert crm.requests[0].url.params["ids"] == "z-1"

    async def test__a_delete_of_many__sends_a_batch_of_ids_per_call(
        self, writes_of: WritesOf
    ) -> None:
        """Zoho takes at most a batch of ids per delete, comma-separated."""
        crm = ScriptedServer(self.outcome("z-1", "z-2"), self.outcome("z-3"))
        writes = writes_of(crm, batch_size=2)

        ids = await writes.delete_many(["z-1", "z-2", "z-3"])

        assert ids == ["z-1", "z-2", "z-3"]
        assert [r.url.params["ids"] for r in crm.requests] == ["z-1,z-2", "z-3"]

    async def test__a_refused_delete__raises_with_zohos_reason(self, writes_of: WritesOf) -> None:
        """Zoho answers a status per id; one not held is refused."""
        refusal = {"code": "INVALID_DATA", "status": "error", "details": {"id": "z-9"}}
        writes = writes_of(ScriptedServer(httpx.Response(200, json={"data": [refusal]})))

        with pytest.raises(WriteError, match="INVALID_DATA"):
            await writes.delete_many(["z-9"])
