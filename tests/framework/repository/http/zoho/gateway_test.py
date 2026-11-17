import json

import httpx

from baseline_common.framework.repository.abstract_repository import AbstractRepository
from baseline_common.framework.repository.http.client.http_client import HttpClient
from baseline_common.framework.repository.http.client.http_config import HttpConfig
from baseline_common.framework.repository.http.gateway import Gateway
from baseline_common.framework.repository.http.zoho.coql_dialect import ZohoCoqlDialect
from baseline_common.framework.repository.http.zoho.write_config import ZohoWriteConfig
from tests.framework.repository.http.client.conftest import ScriptedServer
from tests.framework.repository.http.zoho.conftest import ZohoAccount


class TestGateway:
    @staticmethod
    def gateway(crm: ScriptedServer) -> Gateway[ZohoAccount]:
        """Build a gateway on Zoho's Accounts, read and written by the same object.

        Args:
            crm: The scripted CRM.

        Returns:
            The gateway.
        """
        client = HttpClient(
            "https://crm.test/crm/v8",
            config=HttpConfig(retry_backoff=0.0),
            transport=crm.transport,
        )
        dialect = ZohoCoqlDialect("Accounts")
        config = ZohoWriteConfig(path="Accounts")
        return Gateway(client, ZohoAccount, dialect, config, primary_key="zoho_id")

    def test__the_gateway__fulfills_both_contracts(self) -> None:
        """One object is passed where a caller reads and writes."""
        gateway = self.gateway(ScriptedServer())

        assert isinstance(gateway, AbstractRepository)

    async def test__one_object__reads_then_writes(self, account: ZohoAccount) -> None:
        """A read goes through COQL, a write to the module, on the same client."""
        answer = {"data": [account.model_dump(mode="json", by_alias=True)], "info": {}}
        outcome = {"data": [{"code": "SUCCESS", "status": "success", "details": {"id": "z-1"}}]}
        crm = ScriptedServer(httpx.Response(200, json=answer), httpx.Response(200, json=outcome))
        gateway = self.gateway(crm)

        read = await gateway.get_by_id("z-1")
        updated = await gateway.update_one(account)

        assert read == account
        assert updated == "z-1"
        assert crm.paths() == ["/crm/v8/coql", "/crm/v8/Accounts"]
        assert json.loads(crm.requests[1].content)["data"][0]["id"] == "z-1"
