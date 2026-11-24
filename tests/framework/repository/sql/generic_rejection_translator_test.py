"""Unit tests for the generic translator: the refusal of a dialect no translator reads."""

from sqlalchemy.exc import DataError, DBAPIError, IntegrityError, OperationalError

from baseline_common.framework.repository.sql.abstract_rejection_translator import RejectionReason
from baseline_common.framework.repository.sql.generic_rejection_translator import (
    GenericRejectionTranslator,
)
from baseline_common.framework.repository.sql.sql_write_repository import (
    SqlDialect,
    SqlWriteRepository,
)
from tests.framework.repository.sql.workshop_tables import WorkshopTable


class TestTranslate:
    """Any other dialect is read generically, rather than failing on its error."""

    @staticmethod
    def test__when_a_dialect_has_no_translator__returns_the_generic_translator() -> None:
        """SQLite, say."""
        assert SqlWriteRepository.TRANSLATORS[SqlDialect("sqlite")] is GenericRejectionTranslator

    @staticmethod
    def test__when_a_refusal_is_translated__returns_invalid_value_of_the_written_table() -> None:
        """Only the table is known."""
        translator = GenericRejectionTranslator(WorkshopTable.metadata.tables["machine"])

        rejection = translator.translate(DBAPIError("INSERT", {}, Exception("refused")))

        assert rejection.reason is RejectionReason.INVALID_VALUE
        assert (rejection.table, rejection.constraint) == ("machine", None)

    @staticmethod
    def test__when_errors_differ_in_class__returns_true_for_integrity_and_data_only() -> None:
        """The driver's class decides; a lost connection is no refusal."""
        refused = [
            GenericRejectionTranslator.is_refusal(error_type("INSERT", {}, Exception()))
            for error_type in [IntegrityError, DataError, OperationalError, DBAPIError]
        ]

        assert refused == [True, True, False, False]
