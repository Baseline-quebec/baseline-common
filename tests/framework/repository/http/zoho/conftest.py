"""Fixtures of the Zoho gateway tests: Accounts readers and writers over a scripted CRM."""

from collections.abc import Callable

import pytest
from pydantic import BaseModel, ConfigDict, Field

from baseline_common.framework.repository.http.client.http_client import HttpClient
from baseline_common.framework.repository.http.client.http_config import HttpConfig
from baseline_common.framework.repository.http.read_gateway import ReadGateway
from baseline_common.framework.repository.http.write_gateway import WriteGateway
from baseline_common.framework.repository.http.zoho.coql_dialect import ZohoCoqlDialect
from baseline_common.framework.repository.http.zoho.write_config import ZohoWriteConfig
from tests.framework.repository.http.client.conftest import ScriptedServer


class ZohoAccount(BaseModel):
    """A customer account as Zoho's Accounts module holds it: each alias is a Zoho field."""

    model_config = ConfigDict(frozen=True, validate_by_name=True, validate_by_alias=True)

    zoho_id: str | None = Field(None, alias="id")
    name: str = Field(alias="Account_Name")
    industry: str | None = Field(None, alias="Industry")
    status: str = Field("Actif", alias="Status")
    postal_code: str | None = Field(None, alias="Billing_Code")


# The signature of the `reads_of` fixture, which it and every test using it annotate.
type ReadsOf = Callable[[ScriptedServer], ReadGateway[ZohoAccount]]
# The signature of the `writes_of` fixture, which it and every test using it annotate.
type WritesOf = Callable[..., WriteGateway[ZohoAccount]]


@pytest.fixture
def account() -> ZohoAccount:
    """Return Tremblay et Gagnon as an Accounts record.

    Returns:
        The record.
    """
    return ZohoAccount(
        zoho_id="z-1",
        name="ATELIER TREMBLAY ET GAGNON",
        industry="Machining",
        postal_code="G1K 3X9",
    )


@pytest.fixture
def reads_of() -> ReadsOf:
    """Return a builder of Accounts readers over a scripted CRM.

    Returns:
        The builder: the CRM, then the reader.
    """

    def build(crm: ScriptedServer) -> ReadGateway[ZohoAccount]:
        client = HttpClient(
            "https://crm.test/crm/v8",
            config=HttpConfig(retry_backoff=0.0),
            transport=crm.transport,
        )
        return ReadGateway(client, ZohoAccount, ZohoCoqlDialect("Accounts"), primary_key="zoho_id")

    return build


@pytest.fixture
def writes_of() -> WritesOf:
    """Return a builder of Accounts writers over a scripted CRM.

    Returns:
        The builder: the CRM and the write config's own keys, then the writer.
    """

    def build(crm: ScriptedServer, **config: object) -> WriteGateway[ZohoAccount]:
        client = HttpClient(
            "https://crm.test/crm/v8",
            config=HttpConfig(retry_backoff=0.0),
            transport=crm.transport,
        )
        write_config = ZohoWriteConfig.model_validate({"path": "Accounts", **config})
        return WriteGateway(client, ZohoAccount, write_config)

    return build
