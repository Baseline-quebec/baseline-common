from fastapi import status
from fastapi.testclient import TestClient

from baseline_common.api.settings import Settings
from baseline_common.api.v1.api import ApiV1


class TestHealthRouter:
    def test__health__answers_with_the_injected_settings(self) -> None:
        """Dishka injects into the bound method: the name comes from the container's settings."""
        app = ApiV1(Settings(app_name="probe"))

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["name"] == "probe"

    def test__the_injected_settings__are_not_a_request_parameter(self) -> None:
        """A dependency from the container never shows in the schema as a query parameter."""
        app = ApiV1(Settings(app_name="probe"))

        with TestClient(app) as client:
            schema = client.get("/openapi.json").json()

        assert "parameters" not in schema["paths"]["/health"]["get"]
