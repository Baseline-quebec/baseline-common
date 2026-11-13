"""QdrantRepository — reads and writes documents of a Qdrant collection, through one object."""

from pydantic_ai import Embedder
from qdrant_client import AsyncQdrantClient

from baseline_common.core.repository.abstract_repository import AbstractRepository
from baseline_common.core.repository.vectorstore.qdrant.qdrant_collection import QdrantCollection
from baseline_common.core.repository.vectorstore.qdrant.qdrant_read_repository import (
    QdrantConfig,
    QdrantReadRepository,
)
from baseline_common.core.repository.vectorstore.qdrant.qdrant_write_repository import (
    QdrantWriteRepository,
)
from baseline_common.core.repository.vectorstore.vector_document import VectorDocument


class QdrantRepository[T: VectorDocument](
    QdrantWriteRepository[T], QdrantReadRepository[T], AbstractRepository[T]
):
    """Reads and writes one document type in its Qdrant collection, created on the first write."""

    def __init__(
        self,
        content_type: type[T],
        qdrant_client: AsyncQdrantClient,
        embedding_model: Embedder,
        config: QdrantConfig,
    ) -> None:
        """Bind the collection of the document type, read and written by this same object.

        Args:
            content_type: The document type; it names the collection.
            qdrant_client: The Qdrant async client.
            embedding_model: Embeds the text of a document and the semantic content of a search.
            config: How the collection is built and searched.
        """
        QdrantReadRepository.__init__(self, content_type, qdrant_client, embedding_model, config)
        collection = QdrantCollection(content_type, qdrant_client, config)
        QdrantWriteRepository.__init__(self, self, collection)
