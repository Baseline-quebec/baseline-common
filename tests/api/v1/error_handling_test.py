import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from baseline_common.api.v1.error_handling import RecordErrorMiddleware
from baseline_common.framework.api.error_handling import FrenchValidationError
from baseline_common.framework.repository.abstract_repository import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from tests.api.v1.probe_router import ProbeRouter


class TestRecordErrorMiddleware:
    @staticmethod
    def probe_client(failure: Exception) -> TestClient:
        """A client of an app whose ``/fail`` raises the failure behind the middleware.

        Args:
            failure: What the route raises.

        Returns:
            The client.
        """
        app = FastAPI()
        app.include_router(ProbeRouter(failure))
        app.add_middleware(RecordErrorMiddleware)
        return TestClient(app)

    @pytest.mark.parametrize(
        ("failure", "code", "error"),
        [
            (
                RecordNotFoundError("no record 42"),
                status.HTTP_404_NOT_FOUND,
                "Enregistrement introuvable.",
            ),
            (
                DuplicateRecordError("ids already held: 42"),
                status.HTTP_409_CONFLICT,
                "Un enregistrement porte déjà cet identifiant.",
            ),
            (
                NotImplementedError(),
                status.HTTP_501_NOT_IMPLEMENTED,
                "Cette fonctionnalité n'est pas encore disponible.",
            ),
            (KeyError("x"), status.HTTP_500_INTERNAL_SERVER_ERROR, "Erreur interne du serveur."),
        ],
    )
    def test__an_uncaught_exception__is_answered_in_french_json(
        self, failure: Exception, code: int, error: str
    ) -> None:
        """A known failure answers its own status; anything else answers 500, its cause kept out."""
        client = self.probe_client(failure)

        response = client.get("/fail")

        assert response.status_code == code
        assert response.json() == {"error": error}

    def test__a_successful_request__passes_through(self) -> None:
        """The middleware leaves a response that did not fail untouched."""
        client = self.probe_client(RuntimeError("boom"))

        response = client.get("/echo", params={"count": 3, "ratio": 0.5})

        assert response.json() == {"count": 3, "ratio": 0.5}


@pytest.fixture
def validation_client() -> TestClient:
    """A client of an app answering validation errors with the French handler.

    Returns:
        The client.
    """
    app = FastAPI()
    app.include_router(ProbeRouter(RuntimeError("boom")))
    app.add_exception_handler(RequestValidationError, FrenchValidationError.handle)
    return TestClient(app)


class TestFrenchValidationError:
    def test__one_invalid_field__is_summarised_with_its_location(
        self, validation_client: TestClient
    ) -> None:
        """One error: the summary is its French message, then where it is."""
        response = validation_client.get("/echo", params={"count": 3, "ratio": "half"})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json() == {
            "error": "Requête invalide.",
            "message": "Nombre décimal invalide. (query → ratio)",
            "details": [
                {
                    "location": "query → ratio",
                    "message": "Nombre décimal invalide.",
                    "type": "float_parsing",
                }
            ],
        }

    def test__several_invalid_fields__are_counted_and_listed(
        self, validation_client: TestClient
    ) -> None:
        """Several errors: the summary counts them and names each location."""
        response = validation_client.get("/echo")

        assert response.json()["message"] == "2 champs invalides : query → count, query → ratio."
        assert [detail["message"] for detail in response.json()["details"]] == [
            "Champ requis manquant.",
            "Champ requis manquant.",
        ]
