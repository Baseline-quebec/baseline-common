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

    def test__a_feature_not_built__is_answered_501(self) -> None:
        """The one exception every API shares answers its own status, in French."""
        answer = self.answer(ExceptionHandlerMiddleware, NotImplementedError())

        assert answer == [501, {"error": "Cette fonctionnalité n'est pas encore disponible."}]

    def test__an_unknown_exception__is_answered_500_without_its_cause(self) -> None:
        """Anything else answers 500; its message, maybe data, stays out of the answer."""
        answer = self.answer(ExceptionHandlerMiddleware, KeyError("client 42"))

        assert answer == [500, {"error": "Erreur interne du serveur."}]

    def test__a_subclass__answers_its_own_exceptions_and_the_framework_s(self) -> None:
        """An API adds rows to ``ERRORS``; the framework's stay."""
        assert self.answer(ClosingMiddleware, ClosedError()) == [
            503,
            {"error": "Fermé.", "message": "Revenez demain."},
        ]
        assert self.answer(ClosingMiddleware, NotImplementedError())[0] == 501

    def test__a_narrower_exception__is_answered_as_the_one_it_derives_from(self) -> None:
        """The nearest known class on an exception's MRO answers it."""
        assert self.answer(ClosingMiddleware, ClosedSoonError())[0] == 503


class TestFrenchValidationError:
    def test__a_known_location__is_named_by_its_french_label(self) -> None:
        """A location with a French label reads as its label, not as its path."""
        error = FrenchValidationError([
            {"loc": ("body", "file"), "type": "missing", "msg": "Field required"}
        ])

        response = error.response

        assert json.loads(bytes(response.body))["message"] == "Champ requis manquant. (fichier)"

    def test__an_unknown_type__keeps_the_message_pydantic_gave(self) -> None:
        """A type with no French message falls back on Pydantic's own."""
        error = FrenchValidationError([
            {"loc": ("query", "day"), "type": "date_parsing", "msg": "Bad date"}
        ])

        response = error.response

        assert json.loads(bytes(response.body))["message"] == "Bad date (query → day)"

    def test__another_exception__is_raised_again(self) -> None:
        """The handler only answers validation errors; anything else goes on up."""
        request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})

        with pytest.raises(KeyError):
            FrenchValidationError.handle(request, KeyError("x"))
