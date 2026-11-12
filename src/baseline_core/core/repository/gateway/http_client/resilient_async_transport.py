"""A transport that pools connections and retries idempotent requests on transient failures."""

import asyncio
import contextlib
import random

import httpx
from loguru import logger

from baseline_core.core.repository.gateway.http_client.http_config import HttpConfig


class ResilientAsyncTransport(httpx.AsyncBaseTransport):
    """Async transport that pools connections and retries idempotent requests."""

    def __init__(
        self, config: HttpConfig, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        """Wrap the transport that actually reaches the network.

        Args:
            config: Resilience settings driving pooling and retries.
            transport: The wrapped transport; a pooled ``AsyncHTTPTransport`` when omitted.
        """
        self._config = config
        self._transport = transport or httpx.AsyncHTTPTransport(
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_keepalive_connections,
                keepalive_expiry=config.keepalive_expiry,
            )
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send the request, retrying an idempotent one on a transient failure.

        Args:
            request: The request to send.

        Returns:
            The first answer that is not worth retrying, or the last one once retries run out.
        """
        retryable = request.method in self._config.retry_methods
        attempt = 0

        while True:
            exhausted = not retryable or attempt >= self._config.retry_total

            try:
                response = await self._transport.handle_async_request(request)
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.PoolTimeout,
            ) as error:
                if exhausted:
                    raise

                reason, retry_after = type(error).__name__, None
            else:
                if exhausted or response.status_code not in self._config.retry_statuses:
                    return response

                reason, retry_after = (
                    str(response.status_code),
                    response.headers.get("Retry-After"),
                )
                await response.aclose()

            delay = self.delay(attempt, retry_after)
            # The query is left out: it may carry a key, and this line lands in shared logs.
            logger.warning(
                "{} {} failed ({}), retry {}/{} in {:.1f}s",
                request.method,
                request.url.copy_with(query=None),
                reason,
                attempt + 1,
                self._config.retry_total,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1

    def delay(self, attempt: int, retry_after: str | None) -> float:
        """Compute how long to wait before the next attempt.

        Args:
            attempt: The zero-based attempt that just failed.
            retry_after: The server's ``Retry-After`` header, when it sent one.

        Returns:
            The server's delay in seconds when it gave one, capped; else an exponential backoff.
        """
        if self._config.respect_retry_after and retry_after:
            with contextlib.suppress(ValueError):
                return min(max(float(retry_after), 0.0), self._config.retry_after_cap)

        backoff = self._config.retry_backoff * 2.0**attempt
        return backoff + random.uniform(0, backoff * 0.1)

    async def aclose(self) -> None:
        """Close the wrapped transport and its connection pool."""
        await self._transport.aclose()
