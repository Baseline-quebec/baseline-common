"""A Zoho account, and the signatures of the fixtures that read and write it."""

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from baseline_common.framework.repository.http.read_gateway import ReadGateway
from baseline_common.framework.repository.http.write_gateway import WriteGateway
from tests.framework.repository.http.client.scripted_server import ScriptedServer


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
