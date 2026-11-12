"""ZohoWriteConfig — the shape of Zoho CRM's record API writes."""

from typing import Any

from pydantic import Field, model_validator

from baseline_common.core.repository.gateway.write_gateway import WriteGatewayConfig


class ZohoWriteConfig(WriteGatewayConfig):
    """Zoho's record API: 100 records per call, wrapped in ``data``, each answered by status."""

    upsert_path: str = ""
    """``<path>/upsert`` unless given: a record with an id is updated, another inserted."""

    batch_size: int = 100
    envelope: str = "data"
    primary_key: str = "id"
    delete_ids_param: str = "ids"
    outcome_id: list[str] = Field(default_factory=lambda: ["details", "id"])
    outcome_status: str | None = "status"
    outcome_success: str | None = "success"

    @model_validator(mode="before")
    @classmethod
    def _upsert_under_the_path(cls, data: Any) -> Any:  # noqa: ANN401
        """Default the upsert path to the collection's own, as Zoho lays it out.

        Args:
            data: The config's fields.

        Returns:
            The fields, ``upsert_path`` set when it was not.
        """
        if isinstance(data, dict) and not data.get("upsert_path") and data.get("path"):
            return {**data, "upsert_path": f"{data['path']}/upsert"}

        return data
