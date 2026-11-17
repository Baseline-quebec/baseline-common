"""A machine of a workshop, as a vector document."""

from datetime import datetime
from typing import Annotated, ClassVar

from pydantic import Field

from baseline_common.framework.repository.vectorstore.vector_document import (
    VectorDocument,
    VectorStoreProperty,
)


class Machine(VectorDocument):
    """A machine of a workshop, with a field of each payload index type."""

    __collection_name__: ClassVar[str] = "machines"
    """The collection the machines live in."""

    name: Annotated[
        str,
        VectorStoreProperty(
            filterable=True, embedding_template="{value}", prompt_template="Machine: {value}"
        ),
    ]
    """The machine's name, embedded and prompted."""
    kind: Annotated[str, VectorStoreProperty(filterable=True)] = "lathe"
    """Its kind."""
    workshop_id: Annotated[str | None, VectorStoreProperty(filterable=True)] = None
    """The workshop that houses it; None when none does."""
    tags: Annotated[list[str], VectorStoreProperty(filterable=True)] = Field(default_factory=list)
    """Its tags."""
    hours: Annotated[float, VectorStoreProperty(filterable=True)] = 0.0
    """Hours it has run."""
    shifts: Annotated[int, VectorStoreProperty(filterable=True)] = 0
    """Shifts it has worked."""
    serviced: Annotated[datetime | None, VectorStoreProperty(filterable=True)] = None
    """When it was last serviced; None when never."""
    notes: str = ""
    """Free notes, neither indexed nor embedded."""
