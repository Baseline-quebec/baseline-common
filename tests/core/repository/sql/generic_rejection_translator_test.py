"""Unit tests for the generic translator: the refusal of a dialect no translator reads."""

from sqlalchemy.exc import DataError, DBAPIError, IntegrityError, OperationalError

from baseline_common.core.repository.sql.abstract_rejection_translator import RejectionReason
from baseline_common.core.repository.sql.generic_rejection_translator import (
    GenericRejectionTranslator,
)
from baseline_common.core.repository.sql.sql_write_repository import SqlDialect, SqlWriteRepository
from tests.core.repository.sql.workshop_tables import WorkshopTable


class TestTranslate:
    """Any other dialect is read generically, rather than failing on its error."""

    @staticmethod
    def test__another_dialect__is_read_by_the_generic_translator() -> None:
        """SQLite, say."""
        assert SqlWriteRepository.TRANSLATORS[SqlDialect("sqlite")] is GenericRejectionTranslator

    @staticmethod
    def test__a_refusal__is_an_invalid_value_of_the_written_table() -> None:
        """Only the table is known."""
        translator = GenericRejectionTranslator(WorkshopTable.metadata.tables["machine"])

        rejection = translator.translate(DBAPIError("INSERT", {}, Exception("refused")))

        assert rejection.reason is RejectionReason.INVALID_VALUE
        assert (rejection.table, rejection.constraint) == ("machine", None)

    @staticmethod
    def test__only_integrity_and_data_errors__are_refusals() -> None:
        """The driver's class decides; a lost connection is no refusal."""
        refused = [
            GenericRejectionTranslator.is_refusal(error_type("INSERT", {}, Exception()))
            for error_type in [IntegrityError, DataError, OperationalError, DBAPIError]
        ]

        assert refused == [True, True, False, False]
