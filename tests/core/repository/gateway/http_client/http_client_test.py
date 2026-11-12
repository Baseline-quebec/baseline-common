"""Unit tests for the HTTP client: URL joining, headers, auth and the status policy."""

from collections.abc import Callable

import httpx
import pytest

from baseline_common.core.repository.gateway.http_client.auth import ApiKeyAuth, BearerAuth
from baseline_common.core.repository.gateway.http_client.http_client import HttpClient
from baseline_common.core.repository.gateway.http_client.http_config import HttpConfig
from tests.core.repository.gateway.http_client.conftest import ScriptedServer


class TestRequest:
    """One request: where it goes, what it carries, and what a failure does."""

    async def test__a_path__is_joined_onto_the_base_urls_own_path(
        self, server: Callable[..., ScriptedServer], no_wait: HttpConfig
    ) -> None:
        """``crm/v8`` + ``/Accounts`` is ``crm/v8/Accounts``, whatever the slashes."""
        crm = server(httpx.Response(200))

        async with HttpClient(
            "https://crm.test/crm/v8/", config=no_wait, transport=crm.transport
        ) as client:
            await client.get("/Accounts", params={"page": 2})

        assert crm.paths() == ["/crm/v8/Accounts"]
        assert crm.requests[0].url.params["page"] == "2"

    async def test__a_failing_status__raises_by_default(
        self, server: Callable[..., ScriptedServer], no_wait: HttpConfig
    ) -> None:
        """A 4xx is an error the caller must handle, not a body it might forget to check."""
        erp = server(httpx.Response(404))

        async with HttpClient(
            "https://erp.test", config=no_wait, transport=erp.transport
        ) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get("products/unknown")

    async def test__raising__can_be_turned_off_for_one_call(
        self, server: Callable[..., ScriptedServer], no_wait: HttpConfig
    ) -> None:
        """A caller that expects a 404 reads it as an answer."""
        erp = server(httpx.Response(404))

        async with HttpClient(
            "https://erp.test", config=no_wait, transport=erp.transport
        ) as client:
            response = await client.get("products/unknown", raise_for_status=False)

        assert response.status_code == 404

    @pytest.mark.parametrize(
        ("auth", "header", "value"),
        [
            (ApiKeyAuth("k3y", header="apikey"), "apikey", "k3y"),
            (BearerAuth("t0ken"), "Authorization", "Bearer t0ken"),
        ],
    )
    async def test__every_request__is_signed_and_carries_the_default_headers(
        self,
        server: Callable[..., ScriptedServer],
        no_wait: HttpConfig,
        auth: httpx.Auth,
        header: str,
        value: str,
    ) -> None:
        """The auth and the default headers are set once, on the client."""
        erp = server(httpx.Response(200))

        async with HttpClient(
            "https://erp.test",
            auth=auth,
            headers={"Accept": "application/json"},
            config=no_wait,
            transport=erp.transport,
        ) as client:
            await client.get("products")

        sent = erp.requests[0].headers
        assert sent[header] == value
        assert sent["Accept"] == "application/json"

    @pytest.mark.parametrize("method", ["put", "patch", "delete"])
    async def test__each_verb__sends_its_method(
        self, server: Callable[..., ScriptedServer], no_wait: HttpConfig, method: str
    ) -> None:
        """The shortcuts are ``request`` with their method filled in."""
        crm = server(httpx.Response(204))

        async with HttpClient(
            "https://crm.test", config=no_wait, transport=crm.transport
        ) as client:
            await getattr(client, method)("Accounts/1")

        assert crm.requests[0].method == method.upper()


class TestLifecycle:
    """The client owns a connection pool, and hands it back when closed."""

    async def test__leaving_the_context__closes_the_client(
        self, server: Callable[..., ScriptedServer]
    ) -> None:
        """``async with`` is the whole lifecycle."""
        erp = server(httpx.Response(200))

        async with HttpClient("https://erp.test", transport=erp.transport) as client:
            assert not client.is_closed

        assert client.is_closed
