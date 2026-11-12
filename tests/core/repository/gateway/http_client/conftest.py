"""Doubles for the HTTP client tests: scripted servers standing in for Fidelio and Zoho."""

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import httpx
import pytest

from baseline_core.core.repository.gateway.http_client.auth import OAuthRefreshTokenAuth
from baseline_core.core.repository.gateway.http_client.http_client import HttpClient
from baseline_core.core.repository.gateway.http_client.http_config import HttpConfig


class ScriptedServer:
    """A server answering from a script, and remembering every request it was sent."""

    def __init__(self, *answers: httpx.Response | Exception) -> None:
        """Queue the answers, the last one repeating once the script runs out.

        Args:
            *answers: A response to return, or a transport error to raise, per request.
        """
        self._answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Answer one request from the script.

        Args:
            request: The request the client sent.

        Returns:
            The next scripted response.

        Raises:
            Exception: The next scripted transport error.
        """  # noqa: DOC502 - the scripted error is raised, whatever its type
        self.requests.append(request)
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]

        if isinstance(answer, Exception):
            raise answer

        return answer

    @property
    def transport(self) -> httpx.MockTransport:
        """The transport a client reaches this server through."""
        return httpx.MockTransport(self)

    def paths(self) -> list[str]:
        """Every path the server was asked for, in order."""
        return [request.url.path for request in self.requests]


class Clock:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Read the time.

        Returns:
            The time the test last set.
        """
        return self.now


class ZohoDouble:
    """A Zoho double: an accounts server granting tokens, and a CRM checking them."""

    TOKEN_URL: ClassVar[str] = "https://accounts.test/oauth/v2/token"  # noqa: S105 - a URL
    CLIENT_SECRET: ClassVar[str] = "fake-client-secret"  # noqa: S105 - fake: tests follow it
    REFRESH_TOKEN: ClassVar[str] = "fake-refresh-token"  # noqa: S105 - fake: tests follow it
    API_URL: ClassVar[str] = "https://crm.test/crm/v8"
    SCHEME: ClassVar[str] = "Zoho-oauthtoken"

    def __init__(self, config: HttpConfig, *grants: httpx.Response) -> None:
        """Script the grants, and accept every request signed with a token not marked rejected.

        Args:
            config: The client's resilience settings.
            *grants: The token endpoint's answers, in order.
        """
        self._config = config
        self.accounts = ScriptedServer(*grants)
        self.crm: list[str] = []
        self.rejected: set[str] = set()
        self.clock = Clock()

    @staticmethod
    def grant(value: str, expires_in: int = 3600) -> httpx.Response:
        """Build a token endpoint's answer granting an access token.

        Args:
            value: The access token granted.
            expires_in: Its lifetime in seconds.

        Returns:
            The answer, as Zoho sends it.
        """
        return httpx.Response(200, json={"access_token": value, "expires_in": expires_in})

    def signed(self, value: str) -> str:
        """The ``Authorization`` header a request signed with ``value`` carries."""
        return f"{self.SCHEME} {value}"

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Route a request to the accounts server or the CRM.

        Args:
            request: The request the client sent.

        Returns:
            A grant, or the CRM's answer: 401 for a token marked rejected.
        """
        if request.url == self.TOKEN_URL:
            return self.accounts(request)

        signature = request.headers["Authorization"]
        self.crm.append(signature)

        if signature in self.rejected:
            return httpx.Response(401)

        return httpx.Response(200, json={"data": []})

    def client(self) -> HttpClient:
        """Build a CRM client signing with the refresh-token auth, on this double's clock.

        Returns:
            The client.
        """
        auth = OAuthRefreshTokenAuth(
            self.TOKEN_URL,
            client_id="id",
            client_secret=self.CLIENT_SECRET,
            refresh_token=self.REFRESH_TOKEN,
            scheme=self.SCHEME,
            clock=self.clock,
        )
        return HttpClient(
            self.API_URL,
            auth=auth,
            config=self._config,
            transport=httpx.MockTransport(self.handle),
        )

    def signatures(self) -> list[str]:
        """The ``Authorization`` header of every request the CRM received."""
        return list(self.crm)


@pytest.fixture
def no_wait() -> HttpConfig:
    """Return the default resilience, minus the backoff a test would otherwise sleep through."""
    return HttpConfig(retry_backoff=0.0)


@pytest.fixture
def server() -> Callable[..., ScriptedServer]:
    """Return a factory building a scripted server."""
    return ScriptedServer


@pytest.fixture
def zoho(no_wait: HttpConfig) -> Callable[..., ZohoDouble]:
    """Return a factory building a Zoho double with scripted grants."""
    return partial(ZohoDouble, no_wait)
