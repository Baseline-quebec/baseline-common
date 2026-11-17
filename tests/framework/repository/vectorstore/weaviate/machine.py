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
    __embedding_prefix__: ClassVar[str] = "machine: "

    name: Annotated[
        str,
        VectorStoreProperty(
            searchable=True,
            filterable=True,
            embedding_template="{value}",
            prompt_template="Machine: {value}",
        ),
    ]
    kind: Annotated[str, VectorStoreProperty(filterable=True)] = "lathe"
    mode: Literal["auto", "manual"] = "auto"
    tags: list[str] = Field(default_factory=list)
    hours: float | None = None
    shifts: int | None = None
    active: bool | None = None
    serviced: datetime | None = None
