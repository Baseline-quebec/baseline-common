"""Unit tests for the refresh-token auth: when a token is fetched, reused, and refreshed."""

import asyncio
from collections.abc import Callable
from urllib.parse import parse_qs

import httpx
import pytest

from baseline_common.framework.repository.http.client.auth import (
    OAuthRefreshTokenAuth,
    TokenRefreshError,
)
from tests.framework.repository.http.client.zoho_double import ZohoDouble


class TestGrant:
    """The first request buys a token; the next ones reuse it until it is about to expire."""

    async def test__when_the_first_request_is_sent__returns_a_grant_request_with_a_form_body(
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

    async def test__when_a_token_is_live__returns_one_grant_for_two_requests(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """One grant serves every request until its expiry nears."""
        double = zoho(ZohoDouble.grant("a1"))

        async with double.client() as client:
            await client.get("Accounts")
            double.clock.now = 3000.0
            await client.get("Accounts")

        assert len(double.accounts.requests) == 1
        assert double.signatures() == [double.signed("a1")] * 2

    async def test__when_a_token_is_about_to_expire__returns_a_request_signed_by_a_new_one(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """Inside the margin, the token is renewed before the request rather than after a 401."""
        double = zoho(ZohoDouble.grant("a1"), ZohoDouble.grant("a2"))

        async with double.client() as client:
            await client.get("Accounts")
            double.clock.now = 3590.0
            await client.get("Accounts")

        assert double.signatures() == [double.signed("a1"), double.signed("a2")]

    async def test__when_requests_are_concurrent__returns_one_grant_for_them_all(
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

    async def test__when_a_token_is_rejected__returns_the_answer_to_the_replayed_request(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """The caller never sees the 401."""
        double = zoho(ZohoDouble.grant("a1"), ZohoDouble.grant("a2"))
        double.rejected.add(double.signed("a1"))

        async with double.client() as client:
            response = await client.get("Accounts")

        assert response.status_code == 200
        assert double.signatures() == [double.signed("a1"), double.signed("a2")]

    async def test__when_a_second_401_is_answered__raises_http_status_error(
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

    async def test__when_an_error_body_comes_with_a_200__raises_token_refresh_error(
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

    async def test__when_an_answer_is_not_json__raises_token_refresh_error(
        self, zoho: Callable[..., ZohoDouble]
    ) -> None:
        """An outage page instead of a token says so, rather than crashing on the parse."""
        double = zoho(httpx.Response(503, text="<html>down</html>"))

        async with double.client() as client:
            with pytest.raises(TokenRefreshError, match="HTTP 503"):
                await client.get("Accounts")


class TestSyncClient:
    """The auth guards its cache with an asyncio lock, so a sync client is refused outright."""

    def test__when_a_client_is_sync__raises_not_implemented_error(self) -> None:
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
