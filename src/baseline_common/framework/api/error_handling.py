"""Error handling for an API: a middleware for raised exceptions, a handler for refused requests."""

from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any, ClassVar, override

from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class ErrorResponse(BaseModel):
    """The French answer to an exception: its status, what went wrong, what to do if needed."""

    model_config = ConfigDict(frozen=True)
    """Frozen: an answer never changes once built."""

    status_code: HTTPStatus
    """The HTTP status of the answer."""
    error: str
    """What went wrong, in a few words."""
    message: str | None = None
    """What the user reads beyond the error, if anything."""

    def response(self) -> JSONResponse:
        """Render the French ``{error, message}`` envelope, the message only when there is one.

        Returns:
            The response.
        """
        content = {"error": self.error} | ({"message": self.message} if self.message else {})
        return JSONResponse(status_code=self.status_code, content=content)


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Answer an uncaught exception in French JSON; an API's subclass adds its own to ``ERRORS``."""

    ERRORS: ClassVar[dict[type[Exception], ErrorResponse]] = {
        NotImplementedError: ErrorResponse(
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            error="Cette fonctionnalité n'est pas encore disponible.",
        )
    }
    """The answer to each known exception class, found along an exception's MRO."""
    INTERNAL: ClassVar[ErrorResponse] = ErrorResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR, error="Erreur interne du serveur."
    )
    """The answer to any other exception, its cause kept in the log."""

    @override
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Run the request, translating any uncaught exception to JSON.

        Args:
            request: The inbound request.
            call_next: The rest of the app.

        Returns:
            The app's response, or the answer to the exception it raised.
        """
        try:
            return await call_next(request)
        except Exception as error:  # noqa: BLE001 - the last guard: a request never ends without a body
            return self.respond(error)

    def respond(self, error: Exception) -> Response:
        """Answer an exception: its nearest class in ``ERRORS``, else 500; a subclass adds cases.

        Args:
            error: The exception the request raised.

        Returns:
            The French answer.
        """
        for ancestor in type(error).__mro__:
            if known := self.ERRORS.get(ancestor):
                # The class only: an exception's message may carry the request's data.
                logger.warning("A request ended on {}.", ancestor.__name__)
                return known.response()

        logger.opt(exception=error).error("Unhandled exception during request processing.")
        return self.INTERNAL.response()


class FrenchValidationError(RequestValidationError):
    """A ``RequestValidationError`` rendered as the French ``{error, message, details}``."""

    PYDANTIC_TYPE_FR: ClassVar[dict[str, str]] = {
        "missing": "Champ requis manquant.",
        "extra_forbidden": "Champ non permis.",
        "string_type": "Une chaîne de caractères est attendue.",
        "string_too_short": "La valeur est trop courte.",
        "string_too_long": "La valeur est trop longue.",
        "too_long": "La liste est trop longue.",
        "int_type": "Un entier est attendu.",
        "int_parsing": "Entier invalide.",
        "float_type": "Un nombre décimal est attendu.",
        "float_parsing": "Nombre décimal invalide.",
        "bool_type": "Une valeur booléenne (vrai/faux) est attendue.",
        "bool_parsing": "Valeur booléenne invalide.",
        "url_parsing": "Adresse invalide.",
        "url_scheme": "Adresse invalide : http ou https est attendu.",
        "value_error": "Valeur invalide.",
        "type_error": "Type incorrect.",
        "enum": "Valeur non autorisée.",
        "literal_error": "Valeur non autorisée.",
        "greater_than": "Valeur trop petite.",
        "less_than": "Valeur trop grande.",
        "greater_than_equal": "Valeur trop petite.",
        "less_than_equal": "Valeur trop grande.",
    }
    """The French message of a Pydantic error, keyed by its ``type``."""

    LOCATIONS_FR: ClassVar[dict[str, str]] = {
        "body": "corps de la requête",
        "body.file": "fichier",
        "body.files": "fichiers",
    }
    """The French label of a location, keyed by its dotted path; an API's subclass adds its own."""

    @classmethod
    def handle(cls, _request: Request, exc: Exception) -> JSONResponse:
        """Answer a request FastAPI refused, its errors in French; the app's handler.

        Args:
            _request: The inbound request, unused; the handler signature has it.
            exc: The exception FastAPI routed; anything but a validation error is raised again.

        Returns:
            The French ``{error, message, details}`` envelope, with HTTP 422.
        """
        if not isinstance(exc, RequestValidationError):
            raise exc

        return cls(exc.errors(), body=exc.body).response

    @property
    def response(self) -> JSONResponse:
        """The error as the French JSON envelope: HTTP 422, ``{error, message, details}``."""
        details = [self._translate(err) for err in self.errors()]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "Requête invalide.",
                "message": self._summarise(details),
                "details": details,
            },
        )

    def _translate(self, err: dict[str, Any]) -> dict[str, str]:
        """Normalise one Pydantic error into ``{location, message, type}``.

        Args:
            err: One entry from ``RequestValidationError.errors()``.

        Returns:
            French ``location`` and ``message``, plus the original ``type`` tag.
        """
        path = [str(part) for part in err.get("loc", ())]
        location = self.LOCATIONS_FR.get(".".join(path)) or " → ".join(path) or "(racine)"
        err_type = str(err.get("type", ""))
        message = self.PYDANTIC_TYPE_FR.get(err_type) or str(err.get("msg") or "Valeur invalide.")
        return {"location": location, "message": message, "type": err_type}

    @staticmethod
    def _summarise(details: list[dict[str, str]]) -> str:
        """Build a one-line French summary of the validation errors.

        Args:
            details: Translated validation entries.

        Returns:
            Summary string suitable for direct display to an end user.
        """
        if not details:
            return "Requête invalide."

        if len(details) == 1:
            only = details[0]
            return f"{only['message']} ({only['location']})"

        locations = ", ".join(d["location"] for d in details)
        return f"{len(details)} champs invalides : {locations}."
