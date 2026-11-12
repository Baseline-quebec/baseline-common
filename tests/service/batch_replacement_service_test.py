"""Tests of BatchReplacementService, over Zoho's gateway and over memory."""

import json
from datetime import date
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.repository.gateway.http_client.http_client import HttpClient
from baseline_core.core.repository.gateway.http_client.http_config import HttpConfig
from baseline_core.core.repository.gateway.read_gateway import ReadGateway
from baseline_core.core.repository.gateway.write_gateway import WriteGateway
from baseline_core.core.repository.gateway.zoho.coql_dialect import ZohoCoqlDialect
from baseline_core.core.repository.gateway.zoho.write_config import ZohoWriteConfig
from baseline_core.core.repository.in_memory_repository import InMemoryRepository
from baseline_core.service.batch_replacement_service import (
    BatchReplacement,
    BatchReplacementService,
)
from tests.core.repository.gateway.http_client.conftest import ScriptedServer


class WorkOrder(BaseModel):
    """A machine's task of a week, as a custom Zoho module holds it: each alias a Zoho field."""

    model_config = ConfigDict(frozen=True, validate_by_name=True, validate_by_alias=True)

    zoho_id: str | None = Field(None, alias="id")
    machine: str = Field(alias="Machine")
    week: date = Field(alias="Week")
    task: str = Field(alias="Name")
    hours: float = Field(alias="Hours")


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
    """The service on Zoho's gateway: one COQL read of the scope, then batched writes."""

    @staticmethod
    def service(crm: ScriptedServer) -> BatchReplacementService:
        """Build the service on a Zoho module, the way a project wires it.

        Args:
            crm: The scripted CRM.

        Returns:
            The service, its scope the machine and the week.
        """
        client = HttpClient(
            "https://crm.test/crm/v8",
            config=HttpConfig(retry_backoff=0.0),
            transport=crm.transport,
        )
        reads = ReadGateway(client, WorkOrder, ZohoCoqlDialect("Work_Orders"))
        writes = WriteGateway(client, WorkOrder, ZohoWriteConfig(path="Work_Orders"))
        return BatchReplacementService(reads, writes, ["machine", "week"], "zoho_id")

    @staticmethod
    def held(*ids: str) -> httpx.Response:
        """Answer the COQL query with the work orders already held under the scope.

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

    async def test__the_scope__is_read_with_one_coql_query_sorted_on_the_id(
        self, batch: list[WorkOrder]
    ) -> None:
        """The scope's fields become COQL conditions on their Zoho names."""
        crm = ScriptedServer(httpx.Response(204), self.outcome("z-1", "z-2", "z-3"))

        await self.service(crm).replace(batch)

        query = self.sent(crm.requests[0])["select_query"]
        assert "from Work_Orders where (Machine = 'CNC-2' and Week = '2026-11-09')" in query
        assert "order by id asc" in query

    async def test__a_first_batch__is_inserted(self, batch: list[WorkOrder]) -> None:
        """Nothing held: Zoho answers an empty 204, and the batch is posted in one call."""
        crm = ScriptedServer(httpx.Response(204), self.outcome("z-1", "z-2", "z-3"))

        replacement = await self.service(crm).replace(batch)

        assert replacement == BatchReplacement(
            updated=[], inserted=["z-1", "z-2", "z-3"], deleted=[]
        )
        assert [r.method for r in crm.requests] == ["POST", "POST"]
        assert crm.paths() == ["/crm/v8/coql", "/crm/v8/Work_Orders"]

    async def test__a_larger_batch__rewrites_the_held_ids_and_posts_the_extra(
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

    async def test__a_smaller_batch__deletes_the_records_left_over(
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
    def service(memory: InMemoryRepository[WorkOrder]) -> BatchReplacementService:
        """Build the service on records held in memory.

        Args:
            memory: The records held, read and written.

        Returns:
            The service, its scope the machine and the week.
        """
        return BatchReplacementService(memory, memory, ["machine", "week"], "zoho_id")

    async def test__the_same_batch_sent_twice__leaves_it_once_under_the_same_ids(
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

    async def test__another_scope__is_left_alone(
        self, memory: InMemoryRepository[WorkOrder], batch: list[WorkOrder]
    ) -> None:
        """Only the records of the batch's machine and week are rewritten or deleted."""
        other = batch[0].model_copy(update={"zoho_id": "m-1", "machine": "LATHE-1"})
        await memory.insert_one(other)

        await self.service(memory).replace(batch[:1])

        assert await memory.get_by_id("m-1") == other

    async def test__an_empty_batch__is_refused(
        self, memory: InMemoryRepository[WorkOrder]
    ) -> None:
        """An empty batch names no scope, so it cannot tell which records it replaces."""
        with pytest.raises(ValueError, match="empty batch"):
            await self.service(memory).replace([])

    async def test__a_batch_of_two_scopes__is_refused(
        self, memory: InMemoryRepository[WorkOrder], batch: list[WorkOrder]
    ) -> None:
        """Records of another machine would replace the wrong batch."""
        stray = batch[0].model_copy(update={"machine": "LATHE-1"})

        with pytest.raises(ValueError, match="differ on machine"):
            await self.service(memory).replace([*batch, stray])

    def test__a_scope_without_fields__is_refused(
        self, memory: InMemoryRepository[WorkOrder]
    ) -> None:
        """No field would make every held record part of every batch."""
        with pytest.raises(ValueError, match="at least one field"):
            BatchReplacementService(memory, memory, [], "zoho_id")
