"""Organization — a known organization of the reference system: a client, a supplier, a site."""

from pydantic import BaseModel, ConfigDict


class Organization(BaseModel):
    """A known organization of the reference system."""

    model_config = ConfigDict(frozen=True)
    """Frozen: an organization never changes once read."""

    id_: str
    """Its id in the reference system."""
    name: str
    """Its name, as the reference system writes it."""
    address: str | None = None
    """Its address on one line, when the reference system has one."""
