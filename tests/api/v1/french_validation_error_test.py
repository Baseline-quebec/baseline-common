import json
from functools import partial

import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.requests import Request

from baseline_core.api.settings import Settings
from baseline_core.api.v1.french_validation_error import FrenchValidationError
from tests.api.v1.probe_router import ProbeRouter


@pytest.fixture
def client() -> TestClient:
    """A client of an app answering validation errors with the French handler.

    Returns:
        The client.
    """
    app = FastAPI()
    app.include_router(ProbeRouter(RuntimeError("boom")))
    handler = partial(FrenchValidationError.handle, locations=Settings().validation_locations)
    app.add_exception_handler(RequestValidationError, handler)
    return TestClient(app)


class TestFrenchValidationError:
    def test__one_invalid_field__is_summarised_with_its_location(self, client: TestClient) -> None:
        """One error: the summary is its French message, then where it is."""
        response = client.get("/echo", params={"count": 3, "ratio": "half"})

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

    def test__several_invalid_fields__are_counted_and_listed(self, client: TestClient) -> None:
        """Several errors: the summary counts them and names each location."""
        response = client.get("/echo")

        assert response.json()["message"] == "2 champs invalides : query → count, query → ratio."
        assert [detail["message"] for detail in response.json()["details"]] == [
            "Champ requis manquant.",
            "Champ requis manquant.",
        ]

    def test__a_known_location__is_named_by_its_french_label(self) -> None:
        """A location the settings label reads as its label, not as its path."""
        error = FrenchValidationError(
            [{"loc": ("body", "file"), "type": "missing", "msg": "Field required"}],
            locations=Settings().validation_locations,
        )

        response = error.response

        assert json.loads(bytes(response.body))["message"] == "Champ requis manquant. (fichier)"

    def test__an_unknown_type__keeps_the_message_pydantic_gave(self) -> None:
        """A type with no French message falls back on Pydantic's own."""
        error = FrenchValidationError(
            [{"loc": ("query", "day"), "type": "date_parsing", "msg": "Bad date"}], locations={}
        )

        response = error.response

        assert json.loads(bytes(response.body))["message"] == "Bad date (query → day)"

    def test__another_exception__is_raised_again(self) -> None:
        """The handler only answers validation errors; anything else goes on up."""
        request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})

        with pytest.raises(KeyError):
            FrenchValidationError.handle(request, KeyError("x"), locations={})
