"""A workshop, as a vector document."""

from typing import Annotated, ClassVar

from baseline_common.framework.repository.vectorstore.vector_document import (
    VectorDocument,
    VectorStoreProperty,
)


class Workshop(VectorDocument):
    """A workshop, the parent of the machines it houses."""

    __collection_name__: ClassVar[str] = "workshops"

    name: Annotated[
        str,
        VectorStoreProperty(
            filterable=True, embedding_template="{value}", prompt_template="Workshop: {value}"
        ),
    ]
