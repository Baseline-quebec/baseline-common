"""The example API's errors: the framework's French envelope, and the repositories' own errors."""

from http import HTTPStatus

from baseline_common.framework.api.error_handling import ErrorResponse, ExceptionHandlerMiddleware
from baseline_common.framework.repository.abstract_repository import (
    DuplicateRecordError,
    RecordNotFoundError,
)


class RecordErrorMiddleware(ExceptionHandlerMiddleware):
    """Answer the framework's errors, and a record not held or an id already held."""

    ERRORS = ExceptionHandlerMiddleware.ERRORS | {
        RecordNotFoundError: ErrorResponse(
            status_code=HTTPStatus.NOT_FOUND, error="Enregistrement introuvable."
        ),
        DuplicateRecordError: ErrorResponse(
            status_code=HTTPStatus.CONFLICT, error="Un enregistrement porte déjà cet identifiant."
        ),
    }
    """The framework's answers, and the repositories' errors."""
