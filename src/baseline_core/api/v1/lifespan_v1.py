"""Lifespan of the v1 application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dishka import AsyncContainer
from fastapi import FastAPI
from loguru import logger

from baseline_core.api.api_version import ApiVersion


class LifespanV1:
    """Start the v1 app on its container, and close the container when the app stops."""

    def __init__(self, container: AsyncContainer) -> None:
        """Keep the container the app resolves its dependencies from.

        Args:
            container: The Dishka container; closing it runs its providers' finalizers.
        """
        self.container = container

    @asynccontextmanager
    async def run(self, _app: FastAPI) -> AsyncGenerator[None]:
        """Run the app between its startup and its shutdown.

        Args:
            _app: The app, as FastAPI passes it; the container is already held.

        Yields:
            Once, while the app serves requests.
        """
        logger.info("API version `{}` ready.", ApiVersion.SCHEMA)

        try:
            yield
        finally:
            logger.info("Shutting down API version `{}`.", ApiVersion.SCHEMA)
            await self.container.close()
