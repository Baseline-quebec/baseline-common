"""Unit tests for the resilient transport: what is retried, how often, and how long it waits."""

from collections.abc import Callable
from typing import ClassVar

import httpx
import pytest

from baseline_common.core.repository.gateway.http_client.http_client import HttpClient
from baseline_common.core.repository.gateway.http_client.http_config import HttpConfig
from baseline_common.core.repository.gateway.http_client.resilient_async_transport import (
    ResilientAsyncTransport,
)
from tests.core.repository.gateway.http_client.conftest import ScriptedServer


class TestRetries:
    """Idempotent requests are retried on a transient failure; anything else is sent once."""

    BASE_URL: ClassVar[str] = "https://erp.test"

    async def test__a_busy_server__is_asked_again_until_it_answers(
        self, server: Callable[..., ScriptedServer], no_wait: HttpConfig
    ) -> None:
        """Two 503 then a 200: the caller only ever sees the 200."""
        erp = server(httpx.Response(503), httpx.Response(503), httpx.Response(200))

        async with HttpClient(self.BASE_URL, config=no_wait, transport=erp.transport) as client:
            response = await client.get("products")

        assert response.status_code == 200
        assert len(erp.requests) == 3

    async def test__a_post__is_never_retried(
        self, server: Callable[..., ScriptedServer], no_wait: HttpConfig
    ) -> None:
        """A POST whose answer was lost may have been applied: a second send could duplicate it."""
        crm = server(httpx.Response(503), httpx.Response(201))

        async with HttpClient(self.BASE_URL, config=no_wait, transport=crm.transport) as client:
            response = await client.post("Accounts", json={}, raise_for_status=False)

        assert response.status_code == 503
        assert len(crm.requests) == 1

    async def test__retries_run_out__and_the_last_answer_stands(
        self, server: Callable[..., ScriptedServer]
    ) -> None:
        """A server down for good is asked once plus ``retry_total`` times, then reported."""
        erp = server(httpx.Response(503))
        config = HttpConfig(retry_total=2, retry_backoff=0.0)

        async with HttpClient(self.BASE_URL, config=config, transport=erp.transport) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get("products")

        assert len(erp.requests) == 3

    async def test__a_client_error__is_not_retried(
        self, server: Callable[..., ScriptedServer], no_wait: HttpConfig
    ) -> None:
        """A 404 will be a 404 again: asking twice only spends the rate limit."""
        erp = server(httpx.Response(404))

        async with HttpClient(self.BASE_URL, config=no_wait, transport=erp.transport) as client:
            await client.get("products/unknown", raise_for_status=False)

        assert len(erp.requests) == 1

    async def test__a_dropped_connection__is_retried_for_an_idempotent_request(
        self, server: Callable[..., ScriptedServer], no_wait: HttpConfig
    ) -> None:
        """The connection failing is transient; the request itself was fine."""
        erp = server(httpx.ConnectError("reset"), httpx.Response(200))

        async with HttpClient(self.BASE_URL, config=no_wait, transport=erp.transport) as client:
            response = await client.get("products")

        assert response.status_code == 200
        assert len(erp.requests) == 2

    async def test__a_dropped_connection__surfaces_for_a_post(
        self, server: Callable[..., ScriptedServer], no_wait: HttpConfig
    ) -> None:
        """A non-idempotent request fails loudly rather than being sent a second time."""
        crm = server(httpx.ConnectError("reset"))

        async with HttpClient(self.BASE_URL, config=no_wait, transport=crm.transport) as client:
            with pytest.raises(httpx.ConnectError):
                await client.post("Accounts", json={})

        assert len(crm.requests) == 1


class TestDelay:
    """How long the transport waits before the next attempt."""

    @staticmethod
    def transport(**config: float | bool) -> ResilientAsyncTransport:
        """Build a transport over a server that is never reached.

        Args:
            **config: Overrides of the resilience settings.

        Returns:
            The transport, for its ``delay`` alone.
        """
        return ResilientAsyncTransport(
            HttpConfig.model_validate(config), httpx.MockTransport(lambda _: httpx.Response(200))
        )

    def test__the_servers_retry_after__is_honoured(self) -> None:
        """A rate-limited server says when to come back, and is believed."""
        assert self.transport().delay(0, "7") == 7.0

    def test__a_long_retry_after__is_capped(self) -> None:
        """A server cannot park a worker for an hour."""
        assert self.transport(retry_after_cap=30.0).delay(0, "3600") == 30.0

    def test__an_http_date_retry_after__falls_back_to_the_backoff(self) -> None:
        """The date form is not parsed; the exponential schedule takes over."""
        delay = self.transport(retry_backoff=1.0).delay(2, "Wed, 21 Oct 2026 07:28:00 GMT")

        assert 4.0 <= delay <= 4.4

    def test__the_backoff__doubles_at_every_attempt(self) -> None:
        """Without a hint from the server, each attempt waits about twice the last."""
        transport = self.transport(retry_backoff=1.0, respect_retry_after=False)

        delays = [transport.delay(attempt, None) for attempt in range(3)]

        assert [round(delay) for delay in delays] == [1, 2, 4]
