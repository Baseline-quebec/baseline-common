"""Router for the health-check endpoint."""

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter
from pydantic import BaseModel

from baseline_core.api.settings import Settings


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str = "ok"
    name: str
    """The app answering, as its settings name it."""
    version: str
    """The release answering, to tell which one a deployment runs."""


class HealthRouter(APIRouter):
    """The health check: answers when the app is up, naming the release that runs."""

    def __init__(self, version: str) -> None:
        """Register the endpoint, its dependencies resolved from the Dishka container.

        Args:
            version: The release the endpoint reports.
        """
        super().__init__(tags=["Health"], route_class=DishkaRoute)
        self.version = version
        self.add_api_route("/health", self.health, methods=["GET"])

    async def health(self, settings: FromDishka[Settings]) -> HealthResponse:
        """Report that the app is up.

        Args:
            settings: The app's settings, resolved from the container.

        Returns:
            ``ok``, the app's name and its release.
        """
        return HealthResponse(name=settings.app_name, version=self.version)
