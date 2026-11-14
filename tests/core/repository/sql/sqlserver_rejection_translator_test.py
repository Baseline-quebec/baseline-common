"""Unit tests for the SQL Server translator: what its messages name when the metadata does not."""

import pytest
from sqlalchemy.exc import DBAPIError, OperationalError

from baseline_common.core.repository.sql.abstract_rejection_translator import RejectionReason
from baseline_common.core.repository.sql.sqlserver_rejection_translator import (
    SqlServerRejectionTranslator,
)
from tests.core.repository.sql.workshop_tables import WorkshopTable


@pytest.fixture
def translator() -> SqlServerRejectionTranslator:
    """Reads the refusals of a write to the machines.

    Returns:
        The translator.
    """
    return SqlServerRejectionTranslator(WorkshopTable.metadata.tables["machine"])


class TestTranslate:
    """A message SQLAlchemy's metadata cannot complete still names what it can."""

    @staticmethod
    def refusal(message: str) -> DBAPIError:
        """The error SQLAlchemy raises for a pyodbc error of that message.

        Args:
            message: The message, as SQL Server words it after the driver's prefix.

        Returns:
            The error.
        """
        prefix = "[23000] [Microsoft][ODBC Driver 18 for SQL Server][SQL Server]"
        return DBAPIError("INSERT", {}, Exception("23000", prefix + message))

    @staticmethod
    def test__an_unnamed_primary_key__names_the_columns_of_the_table_primary_key(
        translator: SqlServerRejectionTranslator,
    ) -> None:
        """A generated name; only the first message is read."""
        error = TestTranslate.refusal(
            "Violation of PRIMARY KEY constraint 'PK__machine__3213E83F'. Cannot insert duplicate"
            " key in object 'dbo.machine'. The duplicate key value is (m-1). (2627)"
            " (SQLExecDirectW); [01000] The statement has been terminated. (3621)"
        )

        rejection = translator.translate(error)

        assert rejection.reason is RejectionReason.UNIQUE
        assert (rejection.table, rejection.columns, rejection.values) == (
            "machine",
            ["id"],
            ["m-1"],
        )

    @staticmethod
    def test__an_unknown_foreign_key__names_the_referenced_table(
        translator: SqlServerRejectionTranslator,
    ) -> None:
        """The written table and its columns are unknown without the constraint."""
        error = TestTranslate.refusal(
            'The INSERT statement conflicted with the FOREIGN KEY constraint "FK__part__1". The'
            ' conflict occurred in database "tempdb", table "dbo.machine", column \'id\'. (547)'
            " (SQLExecDirectW)"
        )

        rejection = translator.translate(error)

        assert rejection.reason is RejectionReason.UNKNOWN_REFERENCE
        assert (rejection.table, rejection.other_table, rejection.columns) == (None, "machine", [])

    @staticmethod
    def test__an_error_without_a_number__is_an_invalid_value(
        translator: SqlServerRejectionTranslator,
    ) -> None:
        """Nothing is named."""
        rejection = translator.translate(DBAPIError("INSERT", {}, Exception("lost")))

        assert rejection.reason is RejectionReason.INVALID_VALUE
        assert (rejection.table, rejection.constraint) == (None, None)


class TestIsRefusal:
    """The error's number, or its driver class, tells a refused row from a failure."""

    @staticmethod
    def test__a_truncation__is_refused_though_raised_as_a_syntax_error() -> None:
        """SQLSTATE 42000, error 2628."""
        error = TestTranslate.refusal(
            "String or binary data would be truncated in table 'tempdb.dbo.machine', column"
            " 'serial'. (2628) (SQLExecDirectW)"
        )

        assert SqlServerRejectionTranslator.is_refusal(error)

    @staticmethod
    def test__a_deadlock__is_no_refusal() -> None:
        """Error 1205: the whole transaction is gone, the caller must know."""
        message = (
            "[40001] [Microsoft][ODBC Driver 18 for SQL Server][SQL Server]Transaction (Process"
            " ID 52) was deadlocked on lock resources with another process and has been chosen"
            " as the deadlock victim. Rerun the transaction. (1205) (SQLExecDirectW)"
        )
        error = OperationalError("INSERT", {}, Exception("40001", message))

        assert not SqlServerRejectionTranslator.is_refusal(error)
