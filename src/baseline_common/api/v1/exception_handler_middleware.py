"""Translate uncaught exceptions into structured JSON responses."""

from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import override

from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Convert uncaught exceptions into structured French JSON error responses."""

    @override
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Run the request, translating any uncaught exception to JSON.

        Args:
            request: The inbound request.
            call_next: The rest of the app.

        Returns:
            The app's response; 501 for a feature not yet built; 500 for any other exception.
        """
        try:
            return await call_next(request)

        except NotImplementedError:
            logger.warning("Unimplemented feature requested.")
            return JSONResponse(
                status_code=HTTPStatus.NOT_IMPLEMENTED,
                content={"error": "Cette fonctionnalité n'est pas encore disponible."},
            )

        except Exception:  # noqa: BLE001 - the last guard: a request never ends without a body
            logger.exception("Unhandled exception during request processing.")
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": "Erreur interne du serveur."},
            )
