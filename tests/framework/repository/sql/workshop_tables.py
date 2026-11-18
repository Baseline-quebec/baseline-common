"""The tables of the SQL repository tests: machines, their parts, tools, halls and racks."""

import uuid
from datetime import datetime
from typing import ClassVar

from sqlalchemy import CheckConstraint, DateTime, MetaData, func
from sqlmodel import Field, Relationship, SQLModel


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
    """The test tables' own metadata, its constraints named by convention."""


class Machine(WorkshopTable, table=True):
    """A machine of a site: a unique serial, a size that cannot be negative, a server timestamp."""

    __table_args__ = (CheckConstraint("size >= 0", name="size_non_negative"),)
    """The size cannot be negative."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """The machine's id."""
    serial: str = Field(unique=True, max_length=20)
    """Bounded: SQL Server cannot index text of unbounded length."""
    name: str = "lathe"
    """Its name."""
    size: int = 1
    """Its size, never negative."""
    site: str = ""
    """The site it stands on."""
    created_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={"server_default": func.now(), "nullable": False},
    )
    """When the server stored it; the database fills it in."""


class Part(WorkshopTable, table=True):
    """A part of a machine; the machine cannot be deleted while it has parts."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """The part's id."""
    machine_id: uuid.UUID = Field(foreign_key="machine.id")
    """The machine it belongs to."""


class Tool(WorkshopTable, table=True):
    """A tool, its code unique through a unique index rather than a constraint."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """The tool's id."""
    code: str = Field(index=True, unique=True, max_length=20)
    """Its code, unique through an index."""


class Shift(WorkshopTable, table=True):
    """A machine's shift, keyed by the machine and the shift's rank rather than an id."""

    machine_serial: str = Field(primary_key=True, max_length=20)
    """The serial of the machine, half of the key."""
    rank: int = Field(primary_key=True)
    """The shift's rank, the other half of the key."""


class Hall(WorkshopTable, table=True):
    """A hall of a site; it owns its benches, written and deleted with it."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """The hall's id."""
    site: str = ""
    """The site it stands on."""
    benches: list["Bench"] = Relationship(sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    """The benches it owns."""


class Bench(WorkshopTable, table=True):
    """A bench of a hall; it carries no site of its own."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """The bench's id."""
    hall_id: uuid.UUID | None = Field(default=None, foreign_key="hall.id")
    """The hall that owns it."""
    name: str = ""
    """Its name."""


class Rack(WorkshopTable, table=True):
    """A rack of a site; it owns the racks stacked on it, rows of the same table."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """The rack's id."""
    site: str = ""
    """The site it stands on."""
    base_id: uuid.UUID | None = Field(default=None, foreign_key="rack.id")
    """The rack it is stacked on."""
    stacked: list["Rack"] = Relationship(sa_relationship_kwargs={"cascade": "all"})
    """The racks stacked on it."""
