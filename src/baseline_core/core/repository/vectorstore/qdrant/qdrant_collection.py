"""QdrantCollection — creates the Qdrant collection of a document type, and its indexes."""

import atexit
import os
import tempfile
import types
import typing
from datetime import datetime
from typing import Any, get_args, get_origin

from loguru import logger
from pydantic import BaseModel, ConfigDict, SecretStr
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from baseline_core.core.repository.vectorstore.qdrant.qdrant_read_repository import (
    QdrantConfig,
    QdrantReadRepository,
    QuantizationType,
)
from baseline_core.core.repository.vectorstore.vector_document import (
    VectorDocument,
    VectorStoreProperty,
)


class QdrantConnection(BaseModel):
    """Where a Qdrant server listens, and how a client reaches it."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    host: str = "localhost"
    """The server's host."""

    port: int = 6333
    """The server's HTTP port."""

    grpc_port: int = 6334
    """The server's gRPC port."""

    api_key: SecretStr | None = None
    """The key the server asks for, if any."""

    https: bool = False
    """Whether the connection is encrypted."""

    prefer_grpc: bool = False
    """Whether the client talks gRPC rather than HTTP."""

    tls_cert: str | None = None
    """The PEM certificate the server's TLS is verified against, for a private authority."""

    timeout: int = 60
    """The seconds a request may take."""

    def create_client(self) -> AsyncQdrantClient:
        """Create a client of the server.

        Returns:
            The client; with a certificate, both HTTP and gRPC verify the server against it.
        """
        verify: bool | str = self.https

        if self.tls_cert:
            verify = self._written(self.tls_cert)

        logger.info("Creating a Qdrant client of {}:{}", self.host, self.port)
        return AsyncQdrantClient(
            host=self.host,
            port=self.port,
            grpc_port=self.grpc_port,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            https=self.https,
            prefer_grpc=self.prefer_grpc,
            timeout=self.timeout,
            verify=verify,
        )

    @staticmethod
    def _written(certificate: str) -> str:
        """Write a certificate to a file removed on exit, and point gRPC at it.

        Args:
            certificate: The PEM certificate.

        Returns:
            The file's path, for HTTP to verify against.
        """
        descriptor, path = tempfile.mkstemp(suffix=".pem", prefix="qdrant-ca-")

        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(certificate)

        atexit.register(os.unlink, path)
        # gRPC takes no certificate from the client; it reads its roots from this variable.
        os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = path
        return path


class QdrantCollection:
    """The Qdrant collection of one document type: its vectors, indexes, creation, deletion."""

    def __init__(
        self,
        content_type: type[VectorDocument],
        qdrant_client: AsyncQdrantClient,
        config: QdrantConfig,
    ) -> None:
        """Bind the document type, the client and how the collection is built.

        Args:
            content_type: The document type; it names the collection and gives its indexes.
            qdrant_client: The Qdrant async client.
            config: How the collection is built.
        """
        self._content_type = content_type
        self._client = qdrant_client
        self._config = config
        self._name = content_type.__collection_name__

    @property
    def payload_indexes(self) -> dict[str, PayloadSchemaType]:
        """The payload index of each filterable field, typed from its annotation."""
        indexes = {}

        for field_name, field_info in self._content_type.model_fields.items():
            filterable = any(
                isinstance(metadata, VectorStoreProperty) and metadata.filterable
                for metadata in field_info.metadata
            )

            if filterable:
                indexes[field_name] = self._index_type(field_info.annotation)

        return indexes

    @property
    def quantization(self) -> ScalarQuantization | BinaryQuantization | None:
        """How dense vectors are compressed; None when they are not."""
        if self._config.quantization_type == QuantizationType.SCALAR:
            return ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, always_ram=self._config.quantization_always_ram
                )
            )

        if self._config.quantization_type == QuantizationType.BINARY:
            return BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=self._config.quantization_always_ram)
            )

        return None

    async def create_collection(self, *, recreate_if_exists: bool = False) -> None:
        """Create the collection and its payload indexes, unless it exists.

        Args:
            recreate_if_exists: Whether an existing collection is deleted and created again.
        """
        if recreate_if_exists:
            await self.delete_collection()

        if await self._client.collection_exists(self._name):
            return

        sparse_vectors = (
            {
                QdrantReadRepository.SPARSE: SparseVectorParams(
                    index=SparseIndexParams(on_disk=self._config.on_disk)
                )
            }
            if self._config.use_sparse_vectors
            else None
        )
        await self._client.create_collection(
            collection_name=self._name,
            vectors_config={
                QdrantReadRepository.DENSE: VectorParams(
                    size=self._config.vector_size,
                    distance=self._config.distance_metric,
                    on_disk=self._config.on_disk,
                )
            },
            sparse_vectors_config=sparse_vectors,
            quantization_config=self.quantization,
        )
        await self._create_payload_indexes()
        logger.info(
            "Created Qdrant collection {} ({} quantization)",
            self._name,
            self._config.quantization_type,
        )

    async def ensure_payload_indexes(self) -> None:
        """Give an existing collection the indexes it lacks, e.g. for a field made filterable."""
        if await self._client.collection_exists(self._name):
            await self._create_payload_indexes()

    async def delete_collection(self) -> None:
        """Delete the collection, if it exists."""
        if await self._client.collection_exists(self._name):
            await self._client.delete_collection(self._name)

    async def _create_payload_indexes(self) -> None:
        """Create the payload index of each filterable field."""
        for field_name, index_type in self.payload_indexes.items():
            await self._client.create_payload_index(
                collection_name=self._name, field_name=field_name, field_schema=index_type
            )
            logger.debug("Created a {} index on {}", index_type, field_name)

    @staticmethod
    def _index_type(annotation: Any) -> PayloadSchemaType:  # noqa: ANN401
        """Map a field's annotation to its payload index type.

        Args:
            annotation: The field's type annotation; ``X | None`` indexes as ``X``.

        Returns:
            The index type; a keyword for text, lists and anything not a number or a datetime.
        """
        if get_origin(annotation) in {types.UnionType, typing.Union}:
            annotation = next(arg for arg in get_args(annotation) if arg is not type(None))

        if not isinstance(annotation, type):
            return PayloadSchemaType.KEYWORD

        if issubclass(annotation, datetime):
            return PayloadSchemaType.DATETIME

        # bool before int, which it subclasses: a numeric index would lose its equality.
        if issubclass(annotation, bool):
            return PayloadSchemaType.BOOL

        # An integer index keeps equality; a float one only takes ranges.
        if issubclass(annotation, int):
            return PayloadSchemaType.INTEGER

        if issubclass(annotation, float):
            return PayloadSchemaType.FLOAT

        return PayloadSchemaType.KEYWORD
