"""Tests of BatchReplacementService, over Zoho's gateway and over memory."""

import json
from datetime import date
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from baseline_common.framework.repository.http.client.http_client import HttpClient
from baseline_common.framework.repository.http.client.http_config import HttpConfig
from baseline_common.framework.repository.http.gateway import Gateway
from baseline_common.framework.repository.http.zoho.coql_dialect import ZohoCoqlDialect
from baseline_common.framework.repository.http.zoho.write_config import ZohoWriteConfig
from baseline_common.framework.repository.in_memory_repository import InMemoryRepository
from baseline_common.services.batch_replacement_service import (
    BatchReplacement,
    BatchReplacementService,
)
from tests.framework.repository.http.client.scripted_server import ScriptedServer


class WorkOrder(BaseModel):
    """A machine's task of a week, as a custom Zoho module holds it: each alias a Zoho field."""

    model_config = ConfigDict(frozen=True, validate_by_name=True, validate_by_alias=True)
    """Frozen, and built by field name as well as by Zoho field."""

    zoho_id: str | None = Field(None, alias="id")
    """Zoho's id; None before the order is written."""
    machine: str = Field(alias="Machine")
    """The machine the task is for."""
    week: date = Field(alias="Week")
    """The week of the task."""
    task: str = Field(alias="Name")
    """The task."""
    hours: float = Field(alias="Hours")
    """Hours the task takes."""


@pytest.fixture
def batch() -> list[WorkOrder]:
    """Return the CNC-2's three tasks of the week, as an export sends them.

    Returns:
        The batch, without ids.
    """
    return [
        WorkOrder(machine="CNC-2", week=date(2026, 11, 9), task=task, hours=hours)
        for task, hours in [("Deburr", 2.0), ("Drill", 3.5), ("Polish", 1.0)]
    ]


class TestOverZoho:
    """The service on Zoho's gateway: one COQL read of the records held, then batched writes."""

    @staticmethod
    def service(crm: ScriptedServer) -> BatchReplacementService[WorkOrder]:
        """Build the service on a Zoho module, the way a project wires it.

        Args:
            crm: The scripted CRM.

        Returns:
            The service.
        """
        client = HttpClient(
            "https://crm.test/crm/v8",
            config=HttpConfig(retry_backoff=0.0),
            transport=crm.transport,
        )
        dialect = ZohoCoqlDialect("Work_Orders")
        config = ZohoWriteConfig(path="Work_Orders")
        return BatchReplacementService(
            Gateway(client, WorkOrder, dialect, config, primary_key="zoho_id")
        )

    @staticmethod
    def held(*ids: str) -> httpx.Response:
        """Answer the COQL query with the work orders already held.

        Args:
            *ids: Their ids.

        Returns:
            The answer.
        """
        data = [
            {"id": i, "Machine": "CNC-2", "Week": "2026-11-09", "Name": "Old", "Hours": 1.0}
            for i in ids
        ]
        return httpx.Response(200, json={"data": data, "info": {"more_records": False}})

    @staticmethod
    def outcome(*ids: str) -> httpx.Response:
        """Answer a write as Zoho does when every record succeeds.

        Args:
            *ids: The records' ids.

        Returns:
            The answer.
        """
        data = [{"code": "SUCCESS", "status": "success", "details": {"id": i}} for i in ids]
        return httpx.Response(200, json={"data": data})

    @staticmethod
    def sent(request: httpx.Request) -> dict[str, Any]:
        """Read a request's JSON body.

        Args:
            request: A request to the CRM.

        Returns:
            The body.
        """
        body: dict[str, Any] = json.loads(request.content)
        return body

    async def test__when_the_records_held_are_read__one_coql_query_sorted_on_the_id_is_sent(
        self, batch: list[WorkOrder]
    ) -> None:
        """The held records are read in id order, so the batch reuses their ids in that order."""
        crm = ScriptedServer(httpx.Response(204), self.outcome("z-1", "z-2", "z-3"))

        await self.service(crm).replace(batch)

        query = self.sent(crm.requests[0])["select_query"]
        assert "from Work_Orders" in query
        assert "order by id asc" in query

    async def test__when_nothing_is_held__the_batch_is_inserted(
        self, batch: list[WorkOrder]
    ) -> None:
        """Nothing held: Zoho answers an empty 204, and the batch is posted in one call."""
        crm = ScriptedServer(httpx.Response(204), self.outcome("z-1", "z-2", "z-3"))

        replacement = await self.service(crm).replace(batch)

        assert replacement == BatchReplacement(
            updated=[], inserted=["z-1", "z-2", "z-3"], deleted=[]
        )
        assert [r.method for r in crm.requests] == ["POST", "POST"]
        assert crm.paths() == ["/crm/v8/coql", "/crm/v8/Work_Orders"]

    async def test__when_a_batch_is_larger__rewrites_the_held_ids_and_posts_the_extra(
        self, batch: list[WorkOrder]
    ) -> None:
        """Two held, three sent: two puts carrying the held ids, one post without an id."""
        crm = ScriptedServer(
            self.held("z-1", "z-2"), self.outcome("z-1", "z-2"), self.outcome("z-9")
        )

        replacement = await self.service(crm).replace(batch)

        put, post = self.sent(crm.requests[1])["data"], self.sent(crm.requests[2])["data"]
        assert replacement == BatchReplacement(
            updated=["z-1", "z-2"], inserted=["z-9"], deleted=[]
        )
        assert [r.method for r in crm.requests] == ["POST", "PUT", "POST"]
        assert [(row["id"], row["Name"]) for row in put] == [("z-1", "Deburr"), ("z-2", "Drill")]
        assert post == [{"Machine": "CNC-2", "Week": "2026-11-09", "Name": "Polish", "Hours": 1.0}]

    async def test__when_a_batch_is_smaller__deletes_the_records_left_over(
        self, batch: list[WorkOrder]
    ) -> None:
        """Three held, one sent: one put, then one delete naming every record left over."""
        crm = ScriptedServer(
            self.held("z-1", "z-2", "z-3"), self.outcome("z-1"), self.outcome("z-2", "z-3")
        )

        replacement = await self.service(crm).replace(batch[:1])

        assert replacement == BatchReplacement(
            updated=["z-1"], inserted=[], deleted=["z-2", "z-3"]
        )
        assert [(r.method, r.url.path) for r in crm.requests[1:]] == [
            ("PUT", "/crm/v8/Work_Orders"),
            ("DELETE", "/crm/v8/Work_Orders"),
        ]
        assert crm.requests[2].url.params["ids"] == "z-2,z-3"


@pytest.fixture
def memory() -> InMemoryRepository[WorkOrder]:
    """Return an empty memory of work orders, keyed by their Zoho id.

    Returns:
        The memory.
    """
    return InMemoryRepository(WorkOrder, primary_key="zoho_id")


class TestReplacement:
    """What a replacement leaves held, whatever the backend."""

    @staticmethod
    def service(memory: InMemoryRepository[WorkOrder]) -> BatchReplacementService[WorkOrder]:
        """Build the service on records held in memory.

        Args:
            memory: The records held, read and written.

        Returns:
            The service.
        """
        return BatchReplacementService(memory)

    async def test__when_the_same_batch_is_sent_twice__leaves_it_once_under_the_same_ids(
        self, memory: InMemoryRepository[WorkOrder], batch: list[WorkOrder]
    ) -> None:
        """A rerun of an export rewrites its records rather than doubling them; ids are reused."""
        service = self.service(memory)
        first = await service.replace(batch)

        second = await service.replace(batch)

        held = await memory.get_by_filters(limit=None)
        assert sorted(second.updated) == sorted(first.inserted)
        assert (second.inserted, second.deleted) == ([], [])
        assert sorted(order.task for order in held) == ["Deburr", "Drill", "Polish"]
