"""SqlRepository — reads and writes the rows of one table, through one object."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from baseline_common.core.repository.abstract_repository import AbstractRepository
from baseline_common.core.repository.sql.sql_read_repository import SqlReadRepository
from baseline_common.core.repository.sql.sql_write_repository import SqlWriteRepository


class SqlRepository[T: SQLModel](
    SqlWriteRepository[T], SqlReadRepository[T], AbstractRepository[T]
):
    """Reads and writes the rows of one table, within a fixed scope such as one term."""

    def __init__(
        self,
        session: AsyncSession,
        content_type: type[T],
        id_field: str = "id",
        scope: dict[str, Any] | None = None,
    ) -> None:
        """Bind the session, the table, the column that identifies a row, and the scope.

        Args:
            session: The caller's session; the caller commits it.
            content_type: The SQLModel table class.
            id_field: The column a row's id is read from.
            scope: Column values every row read or written has, e.g. ``{"term_id": term_id}``.
        """
        SqlReadRepository.__init__(self, session, content_type, id_field, scope)
        SqlWriteRepository.__init__(self, self)
