"""WeaviateRepository — reads and writes documents of a Weaviate collection, through one object."""

from pydantic_ai import Embedder
from weaviate import WeaviateAsyncClient

from baseline_common.core.repository.abstract_repository import AbstractRepository
from baseline_common.core.repository.vectorstore.vector_document import VectorDocument
from baseline_common.core.repository.vectorstore.weaviate.weaviate_read_repository import (
    WeaviateConfig,
    WeaviateReadRepository,
)
from baseline_common.core.repository.vectorstore.weaviate.weaviate_write_repository import (
    WeaviateWriteRepository,
)


class WeaviateRepository[T: VectorDocument](
    WeaviateWriteRepository[T], WeaviateReadRepository[T], AbstractRepository[T]
):
    """Reads and writes one document type in its Weaviate collection, for one tenant."""

    def __init__(
        self,
        content_type: type[T],
        weaviate_client: WeaviateAsyncClient,
        embedding_model: Embedder,
        config: WeaviateConfig,
    ) -> None:
        """Bind the collection of the document type, read and written by this same object.

        Args:
            content_type: The document type; it names the collection.
            weaviate_client: The Weaviate async client.
            embedding_model: Embeds the text of a document and the semantic content of a search.
            config: How the collection is searched, and for which tenant.
        """
        WeaviateReadRepository.__init__(
            self, content_type, weaviate_client, embedding_model, config
        )
        WeaviateWriteRepository.__init__(self, self)
