"""GenericRejectionTranslator — a refusal of a database whose errors no translator reads yet."""

from sqlalchemy.exc import DBAPIError

from baseline_common.core.repository.sql.abstract_rejection_translator import (
    AbstractRejectionTranslator,
    RecordRejectedError,
    RejectionReason,
)


class GenericRejectionTranslator(AbstractRejectionTranslator):
    """Names only the table written to: the driver's error is kept as the cause."""

    def translate(self, error: DBAPIError) -> RecordRejectedError:  # noqa: ARG002
        """Read the refusal as an invalid value of the written table.

        Args:
            error: The error raised at flush.

        Returns:
            The refusal.
        """
        return RecordRejectedError(
            reason=RejectionReason.INVALID_VALUE,
            table=self.table.name,
            constraint=None,
            columns=[],
            values=[],
            other_table=None,
        )
