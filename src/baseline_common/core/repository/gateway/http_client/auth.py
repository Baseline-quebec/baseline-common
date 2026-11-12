"""Authentication strategies an HttpClient attaches to every request it sends."""

import asyncio
import time
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

import httpx


class BearerAuth(httpx.Auth):
    """Bearer token authentication."""

    def __init__(self, token: str) -> None:
        """Bind the token.

        Args:
            token: The bearer token.
        """
        self._token = token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        """Attach the ``Authorization`` header to the request.

        Args:
            request: The outgoing request.

        Yields:
            The request, signed.
        """
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


class ApiKeyAuth(httpx.Auth):
    """API key authentication through a configurable header."""

    def __init__(self, key: str, header: str = "X-API-Key") -> None:
        """Bind the key and the header carrying it.

        Args:
            key: The API key.
            header: The header name the server reads the key from.
        """
        self._key = key
        self._header = header

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        """Attach the API key header to the request.

        Args:
            request: The outgoing request.

        Yields:
            The request, signed.
        """
        request.headers[self._header] = self._key
        yield request


class TokenRefreshError(RuntimeError):
    """The authorization server refused to trade the refresh token for an access token."""


class OAuthRefreshTokenAuth(httpx.Auth):
    """OAuth 2.0 refresh-token grant: a long-lived refresh token buys short-lived access tokens."""

    def __init__(  # noqa: PLR0913 - keyword-only after the URL, so no call can swap two secrets
        self,
        token_url: str,
        *,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        scheme: str = "Bearer",
        refresh_margin: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the grant and how the access token is presented.

        Args:
            token_url: The authorization server's token endpoint.
            client_id: The OAuth client id.
            client_secret: The OAuth client secret.
            refresh_token: The long-lived refresh token.
            scheme: The ``Authorization`` scheme; Zoho expects ``Zoho-oauthtoken``.
            refresh_margin: Seconds before expiry at which the token is already refreshed.
            clock: The monotonic clock expiry is measured on.
        """
        self._token_url = token_url
        # Sent as a form body, never as query parameters: a URL ends up in logs and error messages.
        self._grant = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }
        self._scheme = scheme
        self._refresh_margin = refresh_margin
        self._clock = clock
        self._lock = asyncio.Lock()
        self._access_token: str | None = None
        self._expires_at = 0.0

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        """Refuse a sync client: the token cache is guarded by an asyncio lock.

        Args:
            request: The outgoing request.

        Raises:
            NotImplementedError: Always.
        """
        msg = f"{type(self).__name__} only signs requests of an async client"
        raise NotImplementedError(msg)

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Sign the request, refreshing the access token first when it is missing or stale.

        Args:
            request: The outgoing request.

        Yields:
            The token request when one is due, then the request, signed.
        """
        async with self._lock:
            if self._is_stale():
                self._store(await self._read((yield self._token_request())))  # noqa: ASYNC119

        signed_with = self._sign(request)
        response = yield request

        if response.status_code != httpx.codes.UNAUTHORIZED:
            return

        async with self._lock:
            if self._access_token == signed_with:
                self._store(await self._read((yield self._token_request())))  # noqa: ASYNC119

        self._sign(request)
        yield request

    def _is_stale(self) -> bool:
        """Whether the cached access token is missing or about to expire."""
        return (
            self._access_token is None or self._clock() >= self._expires_at - self._refresh_margin
        )

    def _token_request(self) -> httpx.Request:
        """Build the request trading the refresh token for an access token."""
        return httpx.Request("POST", self._token_url, data=self._grant)

    def _sign(self, request: httpx.Request) -> str | None:
        """Attach the cached access token to the request.

        Args:
            request: The outgoing request.

        Returns:
            The token the request was signed with.
        """
        request.headers["Authorization"] = f"{self._scheme} {self._access_token}"
        return self._access_token

    @staticmethod
    async def _read(response: httpx.Response) -> dict[str, Any]:
        """Read the token endpoint's JSON body, whatever its status.

        Args:
            response: The token endpoint's answer.

        Returns:
            The body, or an empty mapping when it is not a JSON object.
        """
        await response.aread()

        try:
            body = response.json()
        except ValueError:
            return {"error": f"HTTP {response.status_code}"}

        return body if isinstance(body, dict) else {}

    def _store(self, body: dict[str, Any]) -> None:
        """Cache the access token a token response carries.

        Args:
            body: The token endpoint's JSON body.

        Raises:
            TokenRefreshError: When the body carries no access token.
        """
        token = body.get("access_token")

        if not isinstance(token, str) or not token:
            msg = f"Token refresh refused: {body.get('error', 'no access_token in the response')}"
            raise TokenRefreshError(msg)

        self._access_token = token
        self._expires_at = self._clock() + float(body.get("expires_in", 3600))
