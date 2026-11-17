"""Fixtures of the repository suite: a workshop's equipment."""

import pytest
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


@pytest.fixture
def press() -> Equipment:
    """Return the press, the only piece given a bay.

    Returns:
        The piece.
    """
    return Equipment(
        id_="m-1",
        name="PRESS ALPHA",
        status="active",
        brand="Haas",
        site=Site(city="Montréal"),
        bay="B2",
        serial="S-001",
    )


@pytest.fixture
def lathe() -> Equipment:
    """Return the lathe, in a city named with a hyphen and accents.

    Returns:
        The piece.
    """
    return Equipment(
        id_="m-4",
        name="LATHE DELTA",
        status="active",
        brand="DMG Mori",
        site=Site(city="Trois-Rivières"),
        serial="S-004",
    )


@pytest.fixture
def inventory(press: Equipment, lathe: Equipment) -> list[Equipment]:
    """Return four pieces: three active, one retired; the press first, the lathe last.

    Args:
        press: The press, first.
        lathe: The lathe, last.

    Returns:
        The pieces, by id.
    """
    drill = Equipment(
        id_="m-2",
        name="DRILL BRAVO",
        status="active",
        brand="Mazak",
        site=Site(city="Québec"),
        serial="S-002",
    )
    saw = Equipment(
        id_="m-3",
        name="SAW CHARLIE",
        status="retired",
        brand="Okuma",
        site=Site(city="Laval"),
        serial="S-003",
    )
    return [press, drill, saw, lathe]
