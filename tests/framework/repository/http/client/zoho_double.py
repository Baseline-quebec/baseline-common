"""A double of Zoho's API, with the clock its tokens expire on."""

from typing import ClassVar

import httpx

from baseline_common.framework.repository.http.client.auth import OAuthRefreshTokenAuth
from baseline_common.framework.repository.http.client.http_client import HttpClient
from baseline_common.framework.repository.http.client.http_config import HttpConfig
from tests.framework.repository.http.client.scripted_server import ScriptedServer


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
    """Where the double grants tokens."""
    CLIENT_SECRET: ClassVar[str] = "fake-client-secret"  # noqa: S105 - fake: tests follow it
    """The client secret the double accepts."""
    REFRESH_TOKEN: ClassVar[str] = "fake-refresh-token"  # noqa: S105 - fake: tests follow it
    """The refresh token the double accepts."""
    API_URL: ClassVar[str] = "https://crm.test/crm/v8"
    """Where the double's CRM answers."""
    SCHEME: ClassVar[str] = "Zoho-oauthtoken"
    """The scheme Zoho puts before a token in ``Authorization``."""

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
