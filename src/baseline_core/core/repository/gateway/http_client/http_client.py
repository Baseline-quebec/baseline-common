"""A configured, resilient async HTTP client scoped to one base URL."""

import time
from collections.abc import Mapping
from typing import Any, Self

import httpx
from loguru import logger

from baseline_core.core.repository.gateway.http_client.http_config import HttpConfig
from baseline_core.core.repository.gateway.http_client.resilient_async_transport import (
    ResilientAsyncTransport,
)


class HttpClient:
    """A configured, resilient async HTTP client scoped to one base URL."""

    def __init__(
        self,
        base_url: str,
        *,
        auth: httpx.Auth | None = None,
        config: HttpConfig | None = None,
        headers: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Bind the base URL, how requests are signed, resilience, headers and transport.

        Args:
            base_url: The base URL request paths are joined onto, its own path kept.
            auth: How every request is signed, if at all.
            config: Resilience settings; the library defaults when omitted.
            headers: Headers merged onto every request.
            transport: The transport reaching the network; tests pass an ``httpx.MockTransport``.
        """
        self._config = config or HttpConfig()
        self._client = httpx.AsyncClient(
            # A trailing slash, so ``crm/v8`` + ``Leads`` joins to ``crm/v8/Leads``.
            base_url=base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(
                connect=self._config.connect_timeout,
                read=self._config.read_timeout,
                write=self._config.write_timeout,
                pool=self._config.pool_timeout,
            ),
            transport=ResilientAsyncTransport(self._config, transport),
            auth=auth,
            headers=headers,
            follow_redirects=True,
        )

    @property
    def is_closed(self) -> bool:
        """Whether the client was closed and can send nothing more."""
        return self._client.is_closed

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        """Send a GET request; see ``request``."""
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        """Send a POST request; see ``request``."""
        return await self.request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        """Send a PUT request; see ``request``."""
        return await self.request("PUT", path, **kwargs)

    async def patch(self, path: str, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        """Send a PATCH request; see ``request``."""
        return await self.request("PATCH", path, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        """Send a DELETE request; see ``request``."""
        return await self.request("DELETE", path, **kwargs)

    async def request(
        self,
        method: str,
        path: str,
        *,
        raise_for_status: bool | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> httpx.Response:
        """Send a request with timeouts, retries and auth applied.

        Args:
            method: The HTTP method.
            path: The path joined onto the base URL, or an absolute URL.
            raise_for_status: Overrides the configured error-raising behavior for this call.
            **kwargs: Passed through to ``httpx.AsyncClient.request`` (params, json, ...).

        Returns:
            The response.

        Raises:
            httpx.HTTPStatusError: On a 4xx or 5xx answer, unless raising is turned off.
        """  # noqa: DOC502 - raised by ``response.raise_for_status``, part of this contract
        started = time.perf_counter()
        response = await self._client.request(method, path.lstrip("/"), **kwargs)
        logger.debug(
            "{} {} -> {} ({:.0f} ms)",
            method,
            response.request.url.copy_with(query=None),
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        should_raise = (
            self._config.raise_for_status if raise_for_status is None else raise_for_status
        )

        if should_raise:
            response.raise_for_status()

        return response

    async def aclose(self) -> None:
        """Close the client and its connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Enter the async context manager.

        Returns:
            The client itself.
        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the client on exit."""
        await self.aclose()
