"""SqlServerRejectionTranslator — reads a refusal from the pyodbc error of SQL Server."""

import re
from enum import IntEnum, StrEnum
from typing import ClassVar, Self

from sqlalchemy import ForeignKeyConstraint, Index, Table
from sqlalchemy.exc import DBAPIError
from sqlalchemy.sql.schema import ColumnCollectionConstraint

from baseline_common.core.repository.sql.abstract_rejection_translator import (
    AbstractRejectionTranslator,
    RecordRejectedError,
    RejectionReason,
)


class SqlServerError(IntEnum):
    """The native error numbers of the refusals SQL Server names; any other number is ``OTHER``."""

    OTHER = 0
    DUPLICATE_KEY = 2627
    DUPLICATE_INDEX_KEY = 2601
    CONFLICT = 547
    NULL_VALUE = 515

    @classmethod
    def _missing_(cls, value: object) -> Self:  # noqa: ARG003
        """Take a number no member has as ``OTHER``.

        Args:
            value: The number.

        Returns:
            ``OTHER``.
        """
        return cls.OTHER


class SqlServerConflict(StrEnum):
    """The kind of constraint error 547 names, worded as in its message."""

    FOREIGN_KEY = "FOREIGN KEY"
    REFERENCE = "REFERENCE"
    CHECK = "CHECK"


class SqlServerRejectionTranslator(AbstractRejectionTranslator):
    """Reads the error number and message; the columns come from the constraint in the metadata."""

    NUMBER: ClassVar[re.Pattern[str]] = re.compile(r"\((?P<number>\d+)\) \(SQL\w+\)")
    KEY_NAME: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:constraint|index) '(?P<constraint>[^']+)'"
    )
    PRIMARY_KEY: ClassVar[re.Pattern[str]] = re.compile(r"PRIMARY KEY constraint '")
    KEY_OBJECT: ClassVar[re.Pattern[str]] = re.compile(r"in object '(?P<table>[^']+)'")
    KEY_VALUES: ClassVar[re.Pattern[str]] = re.compile(
        r"The duplicate key value is \((?P<values>.*)\)\."
    )
    CONFLICT: ClassVar[re.Pattern[str]] = re.compile(
        r"conflicted with the (?:SAME TABLE )?(?P<kind>FOREIGN KEY|REFERENCE|CHECK)"
        r'(?: SAME TABLE)? constraint "(?P<constraint>[^"]+)"'
    )
    CONFLICT_TABLE: ClassVar[re.Pattern[str]] = re.compile(r'table "(?P<table>[^"]+)"')
    NULL_COLUMN: ClassVar[re.Pattern[str]] = re.compile(
        r"column '(?P<column>[^']+)', table '(?P<table>[^']+)'"
    )
    CONFLICT_REASONS: ClassVar[dict[SqlServerConflict, RejectionReason]] = {
        SqlServerConflict.FOREIGN_KEY: RejectionReason.UNKNOWN_REFERENCE,
        SqlServerConflict.REFERENCE: RejectionReason.STILL_REFERENCED,
        SqlServerConflict.CHECK: RejectionReason.CHECK,
    }

    def translate(self, error: DBAPIError) -> RecordRejectedError:
        """Read the refusal from the first message of the pyodbc error SQLAlchemy wraps.

        Args:
            error: The error raised at flush.

        Returns:
            The refusal.
        """
        message = str(error.orig)
        number = self.NUMBER.search(message)
        first_message = message[: number.start()] if number else message

        match SqlServerError(int(number["number"]) if number else 0):
            case SqlServerError.DUPLICATE_KEY | SqlServerError.DUPLICATE_INDEX_KEY:
                return self.duplicate(first_message)

            case SqlServerError.CONFLICT:
                return self.conflict(first_message)

            case SqlServerError.NULL_VALUE:
                return self.null_value(first_message)

            case SqlServerError.OTHER:
                return RecordRejectedError(
                    reason=RejectionReason.INVALID_VALUE,
                    table=None,
                    constraint=None,
                    columns=[],
                    values=[],
                    other_table=None,
                )

    def duplicate(self, message: str) -> RecordRejectedError:
        """A unique constraint, unique index or primary key that a value already holds.

        Args:
            message: The error's first message, errors 2627 and 2601.

        Returns:
            The refusal, with the key's columns and values.
        """
        name = self.KEY_NAME.search(message)
        values = self.KEY_VALUES.search(message)
        object_ = self.KEY_OBJECT.search(message)
        key = self.key_named(name["constraint"]) if name else None
        written = self.table_named(self.unqualified(object_))
        columns = [column.name for column in key.columns] if key is not None else []

        # A primary key left unnamed gets a generated name, which the metadata does not know.
        if key is None and written is not None and self.PRIMARY_KEY.search(message):
            columns = [column.name for column in written.primary_key]

        return RecordRejectedError(
            reason=RejectionReason.UNIQUE,
            table=self.unqualified(object_),
            constraint=name["constraint"] if name else None,
            columns=columns,
            values=values["values"].split(", ") if values else [],
            other_table=None,
        )

    def conflict(self, message: str) -> RecordRejectedError:
        """A foreign key or a check that a statement broke.

        Args:
            message: The error's first message, error 547.

        Returns:
            The refusal, named as PostgreSQL names it: a reference by its referencing table.
        """
        found = self.CONFLICT.search(message)
        named = self.unqualified(self.CONFLICT_TABLE.search(message))
        kind = SqlServerConflict(found["kind"]) if found else SqlServerConflict.CHECK
        key = self.key_named(found["constraint"]) if found else None
        foreign_key = key if isinstance(key, ForeignKeyConstraint) else None
        table: str | None = named
        columns: list[str] = []
        other_table: str | None = None

        # The message names the referenced table on insert, the referencing one on delete.
        if kind is SqlServerConflict.FOREIGN_KEY:
            table = foreign_key.table.name if foreign_key else None
            columns = [column.name for column in foreign_key.columns] if foreign_key else []
            other_table = named

        if kind is SqlServerConflict.REFERENCE:
            columns = (
                [element.column.name for element in foreign_key.elements] if foreign_key else []
            )
            other_table = named

        return RecordRejectedError(
            reason=self.CONFLICT_REASONS[kind],
            table=table,
            constraint=found["constraint"] if found else None,
            columns=columns,
            values=[],
            other_table=other_table,
        )

    def null_value(self, message: str) -> RecordRejectedError:
        """A column left empty although it does not allow nulls.

        Args:
            message: The error's first message, error 515.

        Returns:
            The refusal, with the column.
        """
        found = self.NULL_COLUMN.search(message)
        return RecordRejectedError(
            reason=RejectionReason.REQUIRED,
            table=self.unqualified(found),
            constraint=None,
            columns=[found["column"]] if found else [],
            values=[],
            other_table=None,
        )

    def key_named(self, name: str) -> ColumnCollectionConstraint | Index | None:
        """The constraint or index of that name, in any table of the metadata.

        Args:
            name: The name SQL Server gave.

        Returns:
            The constraint or index, or None when the metadata has none of that name.
        """
        keys = [
            key
            for table in self.table.metadata.tables.values()
            for key in [*table.constraints, *table.indexes]
            if isinstance(key, ColumnCollectionConstraint | Index) and key.name == name
        ]
        return keys[0] if keys else None

    def table_named(self, name: str | None) -> Table | None:
        """The table of that name in the metadata.

        Args:
            name: The table's name, without its schema.

        Returns:
            The table, or None when the metadata has none of that name.
        """
        tables = [table for table in self.table.metadata.tables.values() if table.name == name]
        return tables[0] if tables else None

    @staticmethod
    def unqualified(found: re.Match[str] | None) -> str | None:
        """The table a message names, without its database and schema.

        Args:
            found: The match of a pattern whose ``table`` group holds the name.

        Returns:
            The table's name, or None when the message names none.
        """
        return found["table"].rsplit(".", 1)[-1] if found else None
