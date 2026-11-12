from fastapi import status
from fastapi.testclient import TestClient

from baseline_core.api.settings import Settings
from baseline_core.api.v1.api import ApiV1
from tests.api.v1.probe_router import ProbeRouter


class TestApiV1:
    async def test__the_container__provides_the_settings_the_app_is_built_with(self) -> None:
        """The settings reach Dishka as context, for the routes and the services to resolve."""
        settings = Settings(app_name="probe")
        app = ApiV1(settings)

        resolved = await app.state.dishka_container.get(Settings)

        assert resolved is settings

    def test__a_request_failing_validation__is_answered_in_french(self) -> None:
        """The app registers the French handler: a bad query parameter gets the French envelope."""
        app = ApiV1(Settings(app_name="probe"))
        app.include_router(ProbeRouter(RuntimeError("boom")))

        with TestClient(app) as client:
            response = client.get("/echo", params={"count": "many", "ratio": "0.5"})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["message"] == "Entier invalide. (query → count)"

    def test__a_location_the_project_adds__is_named_by_its_label(self) -> None:
        """The app reads the location labels from its settings: a project adds its own."""
        locations = {"query.count": "nombre de pièces"}
        app = ApiV1(Settings(app_name="probe", validation_locations=locations))
        app.include_router(ProbeRouter(RuntimeError("boom")))

        with TestClient(app) as client:
            response = client.get("/echo", params={"count": "many", "ratio": "0.5"})

        assert response.json()["message"] == "Entier invalide. (nombre de pièces)"

    def test__an_uncaught_exception__is_answered_by_the_middleware(self) -> None:
        """The app registers the middleware: a failing route answers a JSON 500, not a crash."""
        app = ApiV1(Settings(app_name="probe"))
        app.include_router(ProbeRouter(RuntimeError("boom")))

        with TestClient(app) as client:
            response = client.get("/fail")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json() == {"error": "Erreur interne du serveur."}
