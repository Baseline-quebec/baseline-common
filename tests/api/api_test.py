from fastapi import status
from fastapi.testclient import TestClient

from baseline_common.api.api import app
from baseline_common.api.api_version import ApiVersion
from baseline_common.api.settings import get_settings


class TestRootApp:
    def test__the_health_probe__names_the_app_and_its_release(self) -> None:
        """The probe the container polls is unversioned, and names the release that runs."""
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "status": "ok",
            "name": get_settings().app_name,
            "version": ApiVersion.APP,
        }

    def test__the_version__is_mounted_and_serves_its_own_schema(self) -> None:
        """A mount never runs its own lifespan: the root enters it, and the version answers."""
        with TestClient(app) as client:
            response = client.get(f"/{ApiVersion.SCHEMA}/openapi.json")

        assert response.status_code == status.HTTP_200_OK
        assert "/record-matching" in response.json()["paths"]
