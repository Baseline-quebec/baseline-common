"""WriteGateway — writes one record type to an HTTP API, a batch per call."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.repository.abstract_repository import AbstractWriteRepository
from baseline_core.core.repository.gateway.http_client.http_client import HttpClient


class WriteError(RuntimeError):
    """The API refused to write one or more records."""


class WriteGatewayConfig(BaseModel):
    """Where an API takes a record type's writes, and how its requests and answers are shaped."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(min_length=1, description="The collection's path, e.g. ``Accounts``.")
    upsert_path: str = Field(
        min_length=1,
        description="The path taking records to insert, or to update when they carry an id.",
    )
    upsert_body: dict[str, Any] = Field(
        default_factory=dict,
        description="Keys sent beside the records of an upsert, e.g. the fields it matches on.",
    )
    batch_size: int = Field(gt=0, description="Records sent per call, at most.")
    envelope: str = Field(description="The key wrapping the records, sent and answered.")
    primary_key: str = Field(description="The API field holding a record's primary key.")
    delete_ids_param: str = Field(
        min_length=1, description="The query parameter a delete takes its ids in, comma-separated."
    )
    outcome_id: list[str] = Field(
        min_length=1, description="Where, in a record's outcome, the API gives its id."
    )
    outcome_status: str | None = Field(
        None, description="The outcome's status field; None when the API gives none."
    )
    outcome_success: str | None = Field(None, description="The status of a record written.")


class WriteGateway[T: BaseModel](AbstractWriteRepository[T]):
    """Inserts, updates and deletes one record type; the model's aliases are the API's fields."""

    def __init__(
        self, client: HttpClient, content_type: type[T], config: WriteGatewayConfig
    ) -> None:
        """Bind the client, the model and where it is written.

        Args:
            client: The client bound to the API.
            content_type: The model every record is; its aliases are the API's fields.
            config: Where the API takes the writes, and how they are shaped.
        """
        self._client = client
        self.content_type = content_type
        self._config = config

    async def insert_many(self, records: Sequence[T]) -> list[str]:
        """Insert records.

        Args:
            records: The records; their id is left out.

        Returns:
            The ids the API gave them, in order.
        """
        rows = [record.model_dump(mode="json", by_alias=True) for record in records]
        primary_key = self._config.primary_key
        return await self._write(
            self._client.post, [{k: v for k, v in row.items() if k != primary_key} for row in rows]
        )

    async def update_many(self, records: Sequence[T]) -> list[str]:
        """Update records by their id.

        Args:
            records: The records, as they now are.

        Returns:
            Their ids, in order.
        """
        rows = [record.model_dump(mode="json", by_alias=True) for record in records]
        return await self._write(self._client.put, rows)

    async def upsert_many(self, records: Sequence[T]) -> list[str]:
        """Insert records, or update those carrying an id.

        Args:
            records: The records, as they now are.

        Returns:
            Their ids, in order.
        """
        rows = [record.model_dump(mode="json", by_alias=True) for record in records]
        primary_key = self._config.primary_key
        return await self._write(
            self._client.post,
            [{k: v for k, v in row.items() if k != primary_key or v is not None} for row in rows],
            path=self._config.upsert_path,
            body=self._config.upsert_body,
        )

    async def delete_many(self, ids: list[str]) -> list[str]:
        """Delete records, a batch of ids per call.

        Args:
            ids: The records' ids in the API.

        Returns:
            The ids deleted, in order.
        """
        size = self._config.batch_size
        deleted: list[str] = []

        for start in range(0, len(ids), size):
            batch = ",".join(ids[start : start + size])
            response = await self._client.delete(
                self._config.path,
                params={self._config.delete_ids_param: batch},
                raise_for_status=False,
            )
            deleted += self._outcomes(response)

        return deleted

    async def _write(
        self,
        send: Callable[..., Awaitable[httpx.Response]],
        rows: list[dict[str, Any]],
        path: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> list[str]:
        """Send rows, a batch per call.

        Args:
            send: The client's ``post`` to insert, its ``put`` to update.
            rows: The records in the API's fields.
            path: Where to send them; the collection's path by default.
            body: Keys sent beside the records.

        Returns:
            The records' ids, in order.
        """
        size = self._config.batch_size
        ids: list[str] = []

        for start in range(0, len(rows), size):
            response = await send(
                path or self._config.path,
                json={**(body or {}), self._config.envelope: rows[start : start + size]},
                raise_for_status=False,
            )
            ids += self._outcomes(response)

        return ids

    def _outcomes(self, response: httpx.Response) -> list[str]:
        """Read the id of every record a write answered for.

        Args:
            response: The API's answer.

        Returns:
            The ids, in order.

        Raises:
            WriteError: When the API refused a record, with its outcome.
        """
        body = response.json() if response.content else {}

        if self._config.envelope not in body:
            response.raise_for_status()

        outcomes: list[dict[str, Any]] = body.get(self._config.envelope, [])
        status = self._config.outcome_status
        refused = [o for o in outcomes if status and o.get(status) != self._config.outcome_success]

        if refused:
            raise WriteError("; ".join(str(outcome) for outcome in refused))

        return [str(self._id_of(outcome)) for outcome in outcomes]

    def _id_of(self, outcome: dict[str, Any]) -> Any:  # noqa: ANN401
        """Read a record's id from its outcome.

        Args:
            outcome: The API's answer for one record.

        Returns:
            The id.
        """
        value: Any = outcome

        for key in self._config.outcome_id:
            value = value[key]

        return value
