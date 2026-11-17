"""Serve the application."""

from dishka import AsyncContainer, make_async_container
from dishka.integrations.fastapi import FastapiProvider, setup_dishka
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from baseline_common.api.v1.error_handling import ExceptionHandlerMiddleware, FrenchValidationError
from baseline_common.api.v1.lifespan import lifespan as lifespan_v1
from baseline_common.api.v1.routers import record_matching
from baseline_common.assembly.service_provider import ServiceProvider


def init_container() -> AsyncContainer:
    """Create a container for the API.

    Returns:
        A container on the assembly's providers, with FastAPI's request scope wired in.
    """
    return make_async_container(ServiceProvider(), FastapiProvider())


def init_app(container: AsyncContainer) -> FastAPI:
    """Initialize the FastAPI app.

    Args:
        container: The container the routes resolve their services from.

    Returns:
        The versioned app, its routes and French errors in place, its container attached.
    """
    app = FastAPI(lifespan=lifespan_v1)

    app.include_router(record_matching.router)
    app.add_exception_handler(RequestValidationError, FrenchValidationError.handle)
    app.add_middleware(ExceptionHandlerMiddleware)

    setup_dishka(container=container, app=app)
    return app


container = init_container()
app = init_app(container)
