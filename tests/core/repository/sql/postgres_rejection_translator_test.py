"""Unit tests for the PostgreSQL translator: which errors are refused rows."""

import pytest
from sqlalchemy.exc import DBAPIError

from baseline_common.core.repository.sql.postgres_rejection_translator import (
    PostgresRejectionTranslator,
)


class TestIsRefusal:
    """The SQLSTATE's class tells a refused row from a failure of the database."""

    @staticmethod
    def error(sqlstate: str) -> DBAPIError:
        """The error SQLAlchemy raises for an asyncpg error of that SQLSTATE.

        Args:
            sqlstate: The SQLSTATE.

        Returns:
            The error.
        """
        cause = Exception(sqlstate)
        cause.sqlstate = sqlstate  # type: ignore[attr-defined]
        orig = Exception("asyncpg")
        orig.__cause__ = cause
        return DBAPIError("INSERT", {}, orig)

    @staticmethod
    @pytest.mark.parametrize("sqlstate", ["23505", "22001"], ids=["unique", "too-long"])
    def test__an_integrity_or_data_error__is_refused(sqlstate: str) -> None:
        """A key taken twice, a value too long for its column."""
        assert PostgresRejectionTranslator.is_refusal(TestIsRefusal.error(sqlstate))

    @staticmethod
    @pytest.mark.parametrize(
        "sqlstate", ["40P01", "57P01", ""], ids=["deadlock", "shutdown", "none"]
    )
    def test__a_failure_of_the_database__is_no_refusal(sqlstate: str) -> None:
        """A deadlock or a server shutting down stays an error of the database."""
        assert not PostgresRejectionTranslator.is_refusal(TestIsRefusal.error(sqlstate))
