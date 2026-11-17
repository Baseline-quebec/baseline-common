"""A machine of a workshop, as a vector document."""

from datetime import datetime
from typing import Annotated, ClassVar, Literal

from pydantic import Field

from baseline_common.framework.repository.vectorstore.vector_document import (
    VectorDocument,
    VectorStoreProperty,
)


class Machine(VectorDocument):
    """A machine of a workshop, with a field of each Weaviate data type."""

    __collection_name__: ClassVar[str] = "machines"
    """The collection the machines live in."""
    __embedding_prefix__: ClassVar[str] = "machine: "
    """The text put before a machine's content to embed."""

    name: Annotated[
        str,
        VectorStoreProperty(
            searchable=True,
            filterable=True,
            embedding_template="{value}",
            prompt_template="Machine: {value}",
        ),
    ]
    """The machine's name, searched, embedded and prompted."""
    kind: Annotated[str, VectorStoreProperty(filterable=True)] = "lathe"
    """Its kind."""
    mode: Literal["auto", "manual"] = "auto"
    """How it runs."""
    tags: list[str] = Field(default_factory=list)
    """Its tags."""
    hours: float | None = None
    """Hours it has run; None when unknown."""
    shifts: int | None = None
    """Shifts it has worked; None when unknown."""
    active: bool | None = None
    """Whether it is in use; None when unknown."""
    serviced: datetime | None = None
    """When it was last serviced; None when never."""
