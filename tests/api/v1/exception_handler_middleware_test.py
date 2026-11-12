import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from baseline_common.api.v1.exception_handler_middleware import ExceptionHandlerMiddleware
from tests.api.v1.probe_router import ProbeRouter


class TestExceptionHandlerMiddleware:
    @staticmethod
    def client(failure: Exception) -> TestClient:
        """A client of an app whose ``/fail`` raises the failure behind the middleware.

        Args:
            failure: What the route raises.

        Returns:
            The client.
        """
        app = FastAPI()
        app.include_router(ProbeRouter(failure))
        app.add_middleware(ExceptionHandlerMiddleware)
        return TestClient(app)

    @pytest.mark.parametrize(
        ("failure", "code", "error"),
        [
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
        """A feature not yet built answers 501; anything else answers 500, its cause kept out."""
        client = self.client(failure)

        response = client.get("/fail")

        assert response.status_code == code
        assert response.json() == {"error": error}

    def test__a_successful_request__passes_through(self) -> None:
        """The middleware leaves a response that did not fail untouched."""
        client = self.client(RuntimeError("boom"))

        response = client.get("/echo", params={"count": 3, "ratio": 0.5})

        assert response.json() == {"count": 3, "ratio": 0.5}
