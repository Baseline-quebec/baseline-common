"""A machine of a workshop, as a MongoDB document."""

from datetime import datetime

from pydantic import BaseModel, Field

from tests.framework.repository.equipment import Site


class Machine(BaseModel):
    """A machine of a workshop, with a nested field, numbers, a datetime and a list."""

    id_: str | None = None
    name: str
    kind: str = "lathe"
    site: Site
    hours: float = 0.0
    shifts: int = 0
    serviced: datetime | None = None
    tags: list[str] = Field(default_factory=list)
