"""Lifespan function for the FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from loguru import logger

from baseline_common.api.api_version import ApiVersion


if TYPE_CHECKING:
    from dishka import AsyncContainer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Handle the startup and the shutdown of the versioned app.

    Args:
        app: The versioned app, with the Dishka container ``setup_dishka`` attached to it.

    Yields:
        Once, while the app serves requests; the container is closed when it stops.
    """
    container: AsyncContainer = app.state.dishka_container
    logger.info("API version `{}` ready.", ApiVersion.SCHEMA)

    try:
        yield
    finally:
        logger.info("Shutting down API version `{}`.", ApiVersion.SCHEMA)
        await container.close()
