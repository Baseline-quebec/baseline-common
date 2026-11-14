"""PostgresRejectionTranslator — reads a refusal from the asyncpg error of PostgreSQL."""

import re
from enum import StrEnum
from typing import ClassVar, Self, override

from sqlalchemy.exc import DBAPIError

from baseline_common.core.repository.sql.abstract_rejection_translator import (
    AbstractRejectionTranslator,
    RecordRejectedError,
    RejectionReason,
)


class PostgresState(StrEnum):
    """The SQLSTATE codes of the refusals PostgreSQL names; any other code is ``OTHER``."""

    UNIQUE_VIOLATION = "23505"
    FOREIGN_KEY_VIOLATION = "23503"
    CHECK_VIOLATION = "23514"
    NOT_NULL_VIOLATION = "23502"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> Self:  # noqa: ARG003
        """Take a code no member has as ``OTHER``.

        Args:
            value: The code.

        Returns:
            ``OTHER``.
        """
        return cls.OTHER


class SqlStateClass(StrEnum):
    """The SQLSTATE classes of a refused row; any other class is ``OTHER``."""

    DATA_EXCEPTION = "22"
    INTEGRITY_CONSTRAINT_VIOLATION = "23"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> Self:  # noqa: ARG003
        """Take a class no member has as ``OTHER``.

        Args:
            value: The class, the SQLSTATE's first two characters.

        Returns:
            ``OTHER``.
        """
        return cls.OTHER


class PostgresRejectionTranslator(AbstractRejectionTranslator):
    """Reads the SQLSTATE and the detail asyncpg gives, which name the table and the columns."""

    REASONS: ClassVar[dict[PostgresState, RejectionReason]] = {
        PostgresState.UNIQUE_VIOLATION: RejectionReason.UNIQUE,
        PostgresState.FOREIGN_KEY_VIOLATION: RejectionReason.UNKNOWN_REFERENCE,
        PostgresState.CHECK_VIOLATION: RejectionReason.CHECK,
        PostgresState.NOT_NULL_VIOLATION: RejectionReason.REQUIRED,
    }
    KEY_DETAIL: ClassVar[re.Pattern[str]] = re.compile(
        r"Key \((?P<columns>[^)]*)\)=\((?P<values>.*)\)"
    )
    TABLE_DETAIL: ClassVar[re.Pattern[str]] = re.compile(r'table "(?P<table>\w+)"')
    STILL_REFERENCED_DETAIL: ClassVar[re.Pattern[str]] = re.compile(r"still referenced")

    @classmethod
    @override
    def is_refusal(cls, error: DBAPIError) -> bool:
        """Tell a refused row by its SQLSTATE: SQLAlchemy raises asyncpg's data errors as any error.

        Args:
            error: The error raised at flush.

        Returns:
            Whether its SQLSTATE is a data exception or an integrity constraint violation.
        """
        sqlstate = str(getattr(getattr(error.orig, "__cause__", None), "sqlstate", "") or "")
        return SqlStateClass(sqlstate[:2]) is not SqlStateClass.OTHER

    def translate(self, error: DBAPIError) -> RecordRejectedError:
        """Read the refusal from the asyncpg error SQLAlchemy wraps.

        Args:
            error: The error raised at flush.

        Returns:
            The refusal.
        """
        cause = getattr(error.orig, "__cause__", None)
        state = PostgresState(str(getattr(cause, "sqlstate", "") or ""))
        detail = str(getattr(cause, "detail", "") or "")
        key = self.KEY_DETAIL.search(detail)
        other = self.TABLE_DETAIL.search(detail)
        column = getattr(cause, "column_name", None)
        columns = [name.strip() for name in key["columns"].split(",")] if key else []
        reason = self.REASONS.get(state, RejectionReason.INVALID_VALUE)

        if reason is RejectionReason.UNKNOWN_REFERENCE and self.STILL_REFERENCED_DETAIL.search(
            detail
        ):
            reason = RejectionReason.STILL_REFERENCED

        return RecordRejectedError(
            reason=reason,
            table=getattr(cause, "table_name", None),
            constraint=getattr(cause, "constraint_name", None),
            columns=columns or ([column] if column else []),
            values=key["values"].split(", ") if key else [],
            other_table=other["table"] if other else None,
        )
