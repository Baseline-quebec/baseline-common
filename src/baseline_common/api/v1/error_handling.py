"""Error handling for the API: a middleware for raised exceptions, a handler for refused requests."""

from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any, ClassVar, override

from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from baseline_common.framework.repository.abstract_repository import (
    DuplicateRecordError,
    RecordNotFoundError,
)


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Convert uncaught exceptions into structured French JSON error responses."""

    @override
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Run the request, translating any uncaught exception to JSON.

        Args:
            request: The inbound request.
            call_next: The rest of the app.

        Returns:
            The app's response; 404 for a record not held, 409 for an id already held, 501
            for a feature not yet built, 500 for any other exception.
        """
        try:
            return await call_next(request)

        except RecordNotFoundError:
            logger.warning("A request named a record the backend does not hold.")
            return JSONResponse(
                status_code=HTTPStatus.NOT_FOUND, content={"error": "Enregistrement introuvable."}
            )

        except DuplicateRecordError:
            logger.warning("A request gave a record an id the backend already holds.")
            return JSONResponse(
                status_code=HTTPStatus.CONFLICT,
                content={"error": "Un enregistrement porte déjà cet identifiant."},
            )

        except NotImplementedError:
            logger.warning("Unimplemented feature requested.")
            return JSONResponse(
                status_code=HTTPStatus.NOT_IMPLEMENTED,
                content={"error": "Cette fonctionnalité n'est pas encore disponible."},
            )

        except Exception:  # noqa: BLE001 - the last guard: a request never ends without a body
            logger.exception("Unhandled exception during request processing.")
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": "Erreur interne du serveur."},
            )


class FrenchValidationError(RequestValidationError):
    """A ``RequestValidationError`` rendered as the French ``{error, message, details}``."""

    PYDANTIC_TYPE_FR: ClassVar[dict[str, str]] = {
        "missing": "Champ requis manquant.",
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
    """The French label of a location, keyed by its dotted path; a project adds its own."""

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
