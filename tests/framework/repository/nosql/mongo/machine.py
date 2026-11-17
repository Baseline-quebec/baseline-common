"""A machine of a workshop, as a MongoDB document."""

from datetime import datetime

from pydantic import BaseModel, Field

from tests.framework.repository.equipment import Site


class Machine(BaseModel):
    """A machine of a workshop, with a nested field, numbers, a datetime and a list."""

    id_: str | None = None
    """The id MongoDB gives it; None before it is stored."""
    name: str
    """The machine's name."""
    kind: str = "lathe"
    """Its kind."""
    site: Site
    """Where it stands: a nested field."""
    hours: float = 0.0
    """Hours it has run."""
    shifts: int = 0
    """Shifts it has worked."""
    serviced: datetime | None = None
    """When it was last serviced; None when never."""
    tags: list[str] = Field(default_factory=list)
    """Its tags."""
