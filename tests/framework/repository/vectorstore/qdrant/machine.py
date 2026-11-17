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

    name: Annotated[
        str,
        VectorStoreProperty(
            filterable=True, embedding_template="{value}", prompt_template="Machine: {value}"
        ),
    ]
    kind: Annotated[str, VectorStoreProperty(filterable=True)] = "lathe"
    workshop_id: Annotated[str | None, VectorStoreProperty(filterable=True)] = None
    tags: Annotated[list[str], VectorStoreProperty(filterable=True)] = Field(default_factory=list)
    hours: Annotated[float, VectorStoreProperty(filterable=True)] = 0.0
    shifts: Annotated[int, VectorStoreProperty(filterable=True)] = 0
    serviced: Annotated[datetime | None, VectorStoreProperty(filterable=True)] = None
    notes: str = ""
