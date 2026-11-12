"""Unit tests for the refresh-token auth: when a token is fetched, reused, and refreshed."""

import asyncio
from collections.abc import Callable
from urllib.parse import parse_qs

import httpx
import pytest

from baseline_core.core.repository.gateway.http_client.auth import (
    OAuthRefreshTokenAuth,
    TokenRefreshError,
)
from tests.core.repository.gateway.http_client.conftest import ZohoDouble


class TestGrant:
    """The first request buys a token; the next ones reuse it until it is about to expire."""

    async def test__the_refresh_token__is_traded_in_a_form_body(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """The secrets travel in the body, never in a URL a log or an error could print."""
        double = zoho(ZohoDouble.grant("a1"))

        async with double.client() as client:
            await client.get("Accounts")

        (grant,) = double.accounts.requests
        assert grant.method == "POST"
        assert not grant.url.query
        assert parse_qs(grant.content.decode()) == {
            "grant_type": ["refresh_token"],
            "client_id": ["id"],
            "client_secret": [ZohoDouble.CLIENT_SECRET],
            "refresh_token": [ZohoDouble.REFRESH_TOKEN],
        }
        assert double.signatures() == [double.signed("a1")]

    async def test__a_live_token__is_reused(self, zoho: Callable[..., ZohoDouble]) -> None:
        """One grant serves every request until its expiry nears."""
        double = zoho(ZohoDouble.grant("a1"))

        async with double.client() as client:
            await client.get("Accounts")
            double.clock.now = 3000.0
            await client.get("Accounts")

        assert len(double.accounts.requests) == 1
        assert double.signatures() == [double.signed("a1")] * 2

    async def test__a_token_about_to_expire__is_refreshed_first(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """Inside the margin, the token is renewed before the request rather than after a 401."""
        double = zoho(ZohoDouble.grant("a1"), ZohoDouble.grant("a2"))

        async with double.client() as client:
            await client.get("Accounts")
            double.clock.now = 3590.0
            await client.get("Accounts")

        assert double.signatures() == [double.signed("a1"), double.signed("a2")]

    async def test__concurrent_requests__share_one_grant(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """Ten requests in flight buy one token, not ten."""
        double = zoho(ZohoDouble.grant("a1"), ZohoDouble.grant("a2"))

        async with double.client() as client:
            await asyncio.gather(*(client.get("Accounts") for _ in range(10)))

        assert len(double.accounts.requests) == 1
        assert set(double.signatures()) == {double.signed("a1")}


class TestRevocation:
    """A token the server stops accepting early is refreshed once, and the request replayed."""

    async def test__a_rejected_token__is_refreshed_and_the_request_replayed(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """The caller never sees the 401."""
        double = zoho(ZohoDouble.grant("a1"), ZohoDouble.grant("a2"))
        double.rejected.add(double.signed("a1"))

        async with double.client() as client:
            response = await client.get("Accounts")

        assert response.status_code == 200
        assert double.signatures() == [double.signed("a1"), double.signed("a2")]

    async def test__a_second_401__reaches_the_caller(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """One replay only: a token refused twice is a permission problem, not an expiry."""
        double = zoho(ZohoDouble.grant("a1"), ZohoDouble.grant("a2"))
        double.rejected.update({double.signed("a1"), double.signed("a2")})

        async with double.client() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get("Accounts")

        assert len(double.accounts.requests) == 2


class TestRefusedGrant:
    """A refused grant is an error naming the reason, never the secrets."""

    async def test__an_error_body_with_a_200__is_a_refusal(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """Zoho refuses a bad refresh token with a 200: the body decides, not the status."""
        double = zoho(httpx.Response(200, json={"error": "invalid_code"}))

        async with double.client() as client:
            with pytest.raises(TokenRefreshError, match="invalid_code") as refused:
                await client.get("Accounts")

        assert ZohoDouble.CLIENT_SECRET not in str(refused.value)
        assert ZohoDouble.REFRESH_TOKEN not in str(refused.value)
        assert not double.crm

    async def test__a_non_json_answer__is_a_refusal(self, zoho: Callable[..., ZohoDouble]) -> None:
        """An outage page instead of a token says so, rather than crashing on the parse."""
        double = zoho(httpx.Response(503, text="<html>down</html>"))

        async with double.client() as client:
            with pytest.raises(TokenRefreshError, match="HTTP 503"):
                await client.get("Accounts")


class TestSyncClient:
    """The auth guards its cache with an asyncio lock, so a sync client is refused outright."""

    def test__a_sync_client__is_refused(self) -> None:
        """Rather than silently sending the request unsigned."""
        auth = OAuthRefreshTokenAuth(
            ZohoDouble.TOKEN_URL,
            client_id="id",
            client_secret=ZohoDouble.CLIENT_SECRET,
            refresh_token=ZohoDouble.REFRESH_TOKEN,
        )
        transport = httpx.MockTransport(lambda _: httpx.Response(200))

        with (
            httpx.Client(auth=auth, transport=transport) as client,
            pytest.raises(NotImplementedError),
        ):
            client.get(ZohoDouble.API_URL)
