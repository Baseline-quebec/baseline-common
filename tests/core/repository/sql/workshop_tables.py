"""The tables of the SQL repository tests: machines, their parts, and tools."""

import uuid
from datetime import datetime
from typing import ClassVar

from sqlalchemy import CheckConstraint, DateTime, MetaData, func
from sqlmodel import Field, SQLModel


class WorkshopTable(SQLModel):
    """Base of the test tables: a metadata of their own, created inside each test's transaction."""

    metadata: ClassVar[MetaData] = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Machine(WorkshopTable, table=True):
    """A machine of a site: a unique serial, a size that cannot be negative, a server timestamp."""

    __table_args__ = (CheckConstraint("size >= 0", name="size_non_negative"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # SQL Server cannot index text of unbounded length.
    serial: str = Field(unique=True, max_length=20)
    name: str = "lathe"
    size: int = 1
    site: str = ""
    created_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={"server_default": func.now(), "nullable": False},
    )


class Part(WorkshopTable, table=True):
    """A part of a machine; the machine cannot be deleted while it has parts."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    machine_id: uuid.UUID = Field(foreign_key="machine.id")


class Tool(WorkshopTable, table=True):
    """A tool, its code unique through a unique index rather than a constraint."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    code: str = Field(index=True, unique=True, max_length=20)


class Shift(WorkshopTable, table=True):
    """A machine's shift, keyed by the machine and the shift's rank rather than an id."""

    machine_serial: str = Field(primary_key=True, max_length=20)
    rank: int = Field(primary_key=True)
