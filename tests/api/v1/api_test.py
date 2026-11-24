from fastapi import status
from fastapi.testclient import TestClient

from baseline_common.api.v1.api import init_app, init_container
from tests.api.v1.probe_router import ProbeRouter


class TestInitApp:
    def test__when_the_schema_is_fetched__returns_every_route(self, client: TestClient) -> None:
        """Every exposed route reaches the schema, and nothing else does."""
        paths = client.get("/openapi.json").json()["paths"]

        assert set(paths) == {"/record-matching"}

    def test__when_a_request_fails_validation__returns_a_422_in_french(
        self, client: TestClient
    ) -> None:
        """The app registers the French handler: a body missing a field gets the French envelope."""
        response = client.post("/record-matching", json={"candidates": []})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["message"] == "Champ requis manquant. (body → name)"

    def test__when_an_exception_is_uncaught__returns_a_json_500(self) -> None:
        """The app registers the middleware: a failing route answers a JSON 500, not a crash."""
        app = init_app(init_container())
        app.include_router(ProbeRouter(RuntimeError("boom")))

        with TestClient(app) as client:
            response = client.get("/fail")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json() == {"error": "Erreur interne du serveur."}
