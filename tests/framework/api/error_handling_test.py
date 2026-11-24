import json
from http import HTTPStatus

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from baseline_common.framework.api.error_handling import (
    ErrorResponse,
    ExceptionHandlerMiddleware,
    FrenchValidationError,
)


class ClosedError(RuntimeError):
    """A failure an API knows: what a subclass of the middleware adds."""


class ClosedSoonError(ClosedError):
    """A narrower failure, answered as the one it derives from."""


class ClosingMiddleware(ExceptionHandlerMiddleware):
    """The middleware as an API extends it: one exception of its own added."""

    ERRORS = ExceptionHandlerMiddleware.ERRORS | {
        ClosedError: ErrorResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE, error="Fermé.", message="Revenez demain."
        )
    }
    """The framework's answers, and its own."""


class TestExceptionHandlerMiddleware:
    @staticmethod
    def answer(middleware: type[ExceptionHandlerMiddleware], error: Exception) -> list[object]:
        """The status and the body a middleware answers an exception with.

        Args:
            middleware: The middleware class.
            error: The exception.

        Returns:
            ``[status, body]``.
        """
        response = middleware(FastAPI()).respond(error)
        return [response.status_code, json.loads(bytes(response.body))]

    def test__when_a_feature_is_not_built__returns_a_501_in_french(self) -> None:
        """The one exception every API shares answers its own status, in French."""
        answer = self.answer(ExceptionHandlerMiddleware, NotImplementedError())

        assert answer == [501, {"error": "Cette fonctionnalité n'est pas encore disponible."}]

    def test__when_an_exception_is_unknown__returns_a_500_without_its_cause(self) -> None:
        """Anything else answers 500; its message, maybe data, stays out of the answer."""
        answer = self.answer(ExceptionHandlerMiddleware, KeyError("client 42"))

        assert answer == [500, {"error": "Erreur interne du serveur."}]

    def test__when_a_subclass_adds_an_exception__returns_its_own_and_the_framework_s(self) -> None:
        """An API adds rows to ``ERRORS``; the framework's stay."""
        assert self.answer(ClosingMiddleware, ClosedError()) == [
            503,
            {"error": "Fermé.", "message": "Revenez demain."},
        ]
        assert self.answer(ClosingMiddleware, NotImplementedError())[0] == 501

    def test__when_an_exception_is_narrower__returns_the_status_of_its_parent(self) -> None:
        """The nearest known class on an exception's MRO answers it."""
        assert self.answer(ClosingMiddleware, ClosedSoonError())[0] == 503


class TestFrenchValidationError:
    def test__when_a_location_is_known__returns_its_french_label(self) -> None:
        """A location with a French label reads as its label, not as its path."""
        error = FrenchValidationError([
            {"loc": ("body", "file"), "type": "missing", "msg": "Field required"}
        ])

        response = error.response

        assert json.loads(bytes(response.body))["message"] == "Champ requis manquant. (fichier)"

    def test__when_a_type_is_unknown__returns_the_message_pydantic_gave(self) -> None:
        """A type with no French message falls back on Pydantic's own."""
        error = FrenchValidationError([
            {"loc": ("query", "day"), "type": "date_parsing", "msg": "Bad date"}
        ])

        response = error.response

        assert json.loads(bytes(response.body))["message"] == "Bad date (query → day)"

    def test__when_another_exception_is_handled__raises_key_error(self) -> None:
        """The handler only answers validation errors; anything else goes on up."""
        request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})

        with pytest.raises(KeyError):
            FrenchValidationError.handle(request, KeyError("x"))
