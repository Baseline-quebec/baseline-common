"""Root FastAPI app that mounts the versioned sub-apps and, when built, the web UI."""

import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger
from starlette.responses import Response
from starlette.types import Scope

from baseline_common.api.api_version import ApiVersion
from baseline_common.api.settings import Settings
from baseline_common.api.v1.api import ApiV1


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser revalidates on every load, so a changed frontend shows at once."""

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        """Serve the file as usual, telling the browser to check its ETag before reusing it.

        Args:
            full_path: Path of the file to serve.
            stat_result: The file's ``os.stat`` result.
            scope: The ASGI request scope.
            status_code: Status of the response.

        Returns:
            The file response, or a 304 when the browser's copy is current.
        """
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


class Api(FastAPI):
    """The root app: the versioned apps under their schema version, the web UI at the root."""

    def __init__(self, settings: Settings) -> None:
        """Build the versioned apps and mount them, then the web UI when the settings name one.

        Args:
            settings: The configuration of the app and of the versions it mounts.
        """
        super().__init__(title=settings.app_name, version=ApiVersion.APP, lifespan=self.lifespan)
        self.log_level = settings.log_level
        self.v1 = ApiV1(settings)

        self.mount("/" + ApiVersion.SCHEMA, self.v1, name=ApiVersion.SCHEMA)

        if settings.frontend_directory is not None:
            ui = RevalidatedStaticFiles(directory=settings.frontend_directory, html=True)
            self.mount("/", ui, name="frontend")

    @asynccontextmanager
    async def lifespan(self, _app: FastAPI) -> AsyncGenerator[None]:
        """Configure logging, then enter the versioned app's lifespan: a mount never runs its own.

        Args:
            _app: The root app, as FastAPI passes it.

        Yields:
            Once, while the versioned app serves requests.
        """
        logger.remove()
        logger.add(sys.stderr, level=self.log_level)

        async with self.v1.router.lifespan_context(self.v1):
            yield


app = Api(Settings())


if __name__ == "__main__":
    import uvicorn

    settings = Settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")
