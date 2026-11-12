"""FrenchValidationError — a request FastAPI refused, answered in French."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request


class FrenchValidationError(RequestValidationError):
    """A ``RequestValidationError`` rendered as the French ``{error, message, details}``.

    Attributes:
        PYDANTIC_TYPE_FR: Pydantic ``error['type']`` → French message.
    """

    PYDANTIC_TYPE_FR: ClassVar[dict[str, str]] = {
        "missing": "Champ requis manquant.",
        "string_type": "Une chaîne de caractères est attendue.",
        "string_too_short": "La valeur est trop courte.",
        "string_too_long": "La valeur est trop longue.",
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

    def __init__(
        self,
        errors: Sequence[Any],
        *,
        locations: Mapping[str, str],
        body: Any = None,  # noqa: ANN401
    ) -> None:
        """Bind the errors, the French label of each known location, and the refused body.

        Args:
            errors: The entries of ``RequestValidationError.errors()``.
            locations: The French label of a location, keyed by its dotted path (``body.file``).
            body: The body FastAPI refused.
        """
        super().__init__(errors, body=body)
        self.locations = locations

    @classmethod
    def handle(
        cls, _request: Request, exc: Exception, *, locations: Mapping[str, str]
    ) -> JSONResponse:
        """Answer a request FastAPI refused, its errors in French; the app's handler.

        Args:
            _request: The inbound request, unused; the handler signature has it.
            exc: The exception FastAPI routed; anything but a validation error is raised again.
            locations: The French label of a location, keyed by its dotted path.

        Returns:
            The French ``{error, message, details}`` envelope, with HTTP 422.
        """
        if not isinstance(exc, RequestValidationError):
            raise exc

        return cls(exc.errors(), locations=locations, body=exc.body).response

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
        location = self.locations.get(".".join(path)) or " → ".join(path) or "(racine)"
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
