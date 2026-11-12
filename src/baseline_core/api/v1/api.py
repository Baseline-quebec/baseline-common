"""Wire the FastAPI v1 application together."""

from functools import partial

from dishka import make_async_container
from dishka.integrations.fastapi import FastapiProvider, setup_dishka
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from baseline_core.api.api_version import ApiVersion
from baseline_core.api.settings import Settings
from baseline_core.api.v1.container.service_provider import ServiceProvider
from baseline_core.api.v1.exception_handler_middleware import ExceptionHandlerMiddleware
from baseline_core.api.v1.french_validation_error import FrenchValidationError
from baseline_core.api.v1.lifespan_v1 import LifespanV1
from baseline_core.api.v1.routers.health_router import HealthRouter


class ApiV1(FastAPI):
    """The v1 app: its routes, its French errors, and the Dishka container they resolve from."""

    def __init__(self, settings: Settings) -> None:
        """Build the container from the settings, then the app on it.

        Args:
            settings: The configuration the app and its services are built with.
        """
        container = make_async_container(
            FastapiProvider(), ServiceProvider(), context={Settings: settings}
        )
        super().__init__(title=f"{settings.app_name} v1", lifespan=LifespanV1(container).run)

        self.include_router(HealthRouter(ApiVersion.APP))
        self.add_exception_handler(
            RequestValidationError,
            partial(FrenchValidationError.handle, locations=settings.validation_locations),
        )
        self.add_middleware(ExceptionHandlerMiddleware)
        setup_dishka(container=container, app=self)
