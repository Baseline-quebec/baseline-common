"""A piece of equipment and the site it stands on."""

from pydantic import BaseModel, ConfigDict


class Site(BaseModel):
    """Where a piece of equipment stands."""

    model_config = ConfigDict(frozen=True)

    city: str


class Equipment(BaseModel):
    """A piece of a workshop's equipment, with a nested field and optional ones."""

    model_config = ConfigDict(frozen=True)

    id_: str | None = None
    name: str
    status: str
    brand: str | None = None
    site: Site
    bay: str | None = None
    serial: str
