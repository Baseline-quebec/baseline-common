"""A piece of equipment and the site it stands on."""

from pydantic import BaseModel, ConfigDict


class Site(BaseModel):
    """Where a piece of equipment stands."""

    model_config = ConfigDict(frozen=True)
    """Frozen, so a record compares by value."""

    city: str
    """The city the site is in."""


class Equipment(BaseModel):
    """A piece of a workshop's equipment, with a nested field and optional ones."""

    model_config = ConfigDict(frozen=True)
    """Frozen, so a record compares by value."""

    id_: str | None = None
    """The id the store gives it; None before it is stored."""
    name: str
    """The equipment's name."""
    status: str
    """Its status, such as ``active``."""
    brand: str | None = None
    """Its brand; None when unknown."""
    site: Site
    """Where it stands: a nested field."""
    bay: str | None = None
    """The bay it stands in; None when unknown."""
    serial: str
    """Its serial number."""
