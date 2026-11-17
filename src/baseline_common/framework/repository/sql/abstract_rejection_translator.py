"""AbstractRejectionTranslator — reads a write the database refused from its driver's error."""

from abc import ABC, abstractmethod
from enum import StrEnum

from sqlalchemy import Table
from sqlalchemy.exc import DataError, DBAPIError, IntegrityError


class RejectionReason(StrEnum):
    """Which kind of constraint refused the write."""

    UNIQUE = "unique"
    UNKNOWN_REFERENCE = "unknown_reference"
    STILL_REFERENCED = "still_referenced"
    CHECK = "check"
    REQUIRED = "required"
    INVALID_VALUE = "invalid_value"


class RecordRejectedError(ValueError):
    """A write the database refused, with the constraint, columns and values it named."""

    def __init__(
        self,
        reason: RejectionReason,
        table: str | None,
        constraint: str | None,
        columns: list[str],
        values: list[str],
        other_table: str | None,
    ) -> None:
        """Bind what the database said about the refusal.

        Args:
            reason: The kind of constraint.
            table: The table written to.
            constraint: The constraint's name.
            columns: The columns the constraint names, or the column a value was missing in.
            values: The values of those columns, when the database gave them.
            other_table: The table a foreign key points to, or is pointed at from.
        """
        super().__init__(f"{reason} on {table}: {constraint}")
        self.reason = reason
        self.table = table
        self.constraint = constraint
        self.columns = columns
        self.values = values
        self.other_table = other_table


class AbstractRejectionTranslator(ABC):
    """Reads a refusal as one database server words it; one implementation per SQL dialect."""

    def __init__(self, table: Table) -> None:
        """Bind the table written to; its metadata names the constraints of every table.

        Args:
            table: The table the repository writes.
        """
        self.table = table

    @classmethod
    def is_refusal(cls, error: DBAPIError) -> bool:
        """Tell a refused row from a failure of the database, e.g. a deadlock or a lost connection.

        Args:
            error: The error raised at flush.

        Returns:
            Whether the driver raised it as an integrity or data error.
        """
        return isinstance(error, IntegrityError | DataError)

    @abstractmethod
    def translate(self, error: DBAPIError) -> RecordRejectedError:
        """Read the refusal from the driver's error SQLAlchemy wraps.

        Args:
            error: The error raised at flush.

        Returns:
            The refusal.
        """
