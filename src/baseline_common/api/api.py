"""Root FastAPI app that mounts the versioned sub-apps."""

import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from baseline_common.api.api_version import ApiVersion
from baseline_common.api.settings import get_api_settings
from baseline_common.api.v1.api import app as app_v1
from baseline_common.api.v1.lifespan import lifespan as lifespan_v1


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str = "ok"
    """Always ``ok``: an app that answers is up."""
    name: str
    """The app answering, as its settings name it."""
    version: str
    """The release answering, to tell which one a deployment runs."""


@asynccontextmanager
async def root_lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Configure logging, then manually enter the versioned lifespan before yielding.

    Args:
        _app: The root app, which owns no dependencies of its own.

    Yields:
        Once the mounted version is started up: a mounted sub-app never runs its own lifespan.
    """
    logger.remove()
    logger.add(sys.stderr, level=get_api_settings().log_level)

    async with lifespan_v1(app_v1):
        yield


app = FastAPI(title=get_api_settings().app_name, version=ApiVersion.APP, lifespan=root_lifespan)

app.mount("/" + ApiVersion.SCHEMA, app_v1, name=ApiVersion.SCHEMA)


@app.get("/health")
def health() -> HealthResponse:
    """Answer the container's health probe, naming the app and the release that runs.

    Returns:
        ``ok``, the app's name and its release.
    """
    return HealthResponse(name=get_api_settings().app_name, version=ApiVersion.APP)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host=get_api_settings().api_host, port=get_api_settings().api_port, log_level="info"
    )
