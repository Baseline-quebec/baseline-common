"""Unit tests for the Qdrant collection: its payload indexes, vectors, and the client's TLS."""

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from qdrant_client.models import (
    BinaryQuantization,
    PayloadSchemaType,
    QuantizationSearchParams,
    ScalarQuantization,
    SearchParams,
)

from baseline_common.framework.repository.vectorstore.qdrant import qdrant_collection
from baseline_common.framework.repository.vectorstore.qdrant.qdrant_collection import (
    QdrantCollection,
    QdrantConnection,
)
from baseline_common.framework.repository.vectorstore.qdrant.qdrant_read_repository import (
    QdrantConfig,
    QdrantReadRepository,
    QuantizationType,
)
from baseline_common.framework.repository.vectorstore.vector_document import (
    VectorDocument,
    VectorStoreProperty,
)
from tests.framework.repository.vectorstore.qdrant.machine import Machine


class Gauge(VectorDocument):
    """A gauge, with optional and boolean fields."""

    __collection_name__: ClassVar[str] = "gauges"
    """The collection the gauges live in."""

    calibrated: Annotated[bool, VectorStoreProperty(filterable=True)] = False
    """Whether the gauge is calibrated."""
    cycles: Annotated[int | None, VectorStoreProperty(filterable=True)] = None
    """Cycles it has run; None when unknown."""
    reading: Annotated[float | None, VectorStoreProperty(filterable=True)] = None
    """Its last reading; None when unknown."""
    label: str = ""
    """Its label, not indexed."""


class TestPayloadIndexes:
    """Each filterable field gets the index its annotation calls for."""

    @staticmethod
    def test__when_fields_have_different_types__returns_the_index_of_each_type(
        absent: AsyncMock, config: QdrantConfig
    ) -> None:
        """Text and lists by keyword, numbers by range, a datetime by instant."""
        collection = QdrantCollection(Machine, absent, config)

        assert collection.payload_indexes == {
            "id_": PayloadSchemaType.KEYWORD,
            "name": PayloadSchemaType.KEYWORD,
            "kind": PayloadSchemaType.KEYWORD,
            "workshop_id": PayloadSchemaType.KEYWORD,
            "tags": PayloadSchemaType.KEYWORD,
            "hours": PayloadSchemaType.FLOAT,
            "shifts": PayloadSchemaType.INTEGER,
            "serviced": PayloadSchemaType.DATETIME,
        }

    @staticmethod
    def test__when_a_field_is_a_boolean__returns_a_boolean_index(
        absent: AsyncMock, config: QdrantConfig
    ) -> None:
        """A bool subclasses int; an optional field indexes as its type; a label is not indexed."""
        collection = QdrantCollection(Gauge, absent, config)

        assert collection.payload_indexes == {
            "id_": PayloadSchemaType.KEYWORD,
            "calibrated": PayloadSchemaType.BOOL,
            "cycles": PayloadSchemaType.INTEGER,
            "reading": PayloadSchemaType.FLOAT,
        }

    @staticmethod
    def test__when_a_datetime_is_optional__returns_a_datetime_index(
        absent: AsyncMock, config: QdrantConfig
    ) -> None:
        """X | None indexes as X."""

        class Stamp(VectorDocument):
            __collection_name__: ClassVar[str] = "stamps"
            """The collection the stamps live in."""

            at: Annotated[datetime | None, VectorStoreProperty(filterable=True)] = None
            """When it was stamped; None when unknown."""

        assert QdrantCollection(Stamp, absent, config).payload_indexes == {
            "id_": PayloadSchemaType.KEYWORD,
            "at": PayloadSchemaType.DATETIME,
        }


class TestCreation:
    """A collection is created once, with its vectors, compression and indexes."""

    @staticmethod
    async def test__when_a_collection_is_new__returns_the_vectors_and_indexes_created(
        absent: AsyncMock,
    ) -> None:
        """A dense and a sparse vector, scalar compression, one index per filterable field."""
        config = QdrantConfig(
            top_k=3, vector_size=8, quantization_type=QuantizationType.SCALAR, on_disk=True
        )
        collection = QdrantCollection(Gauge, absent, config)

        await collection.create_collection()

        created = absent.create_collection.call_args.kwargs
        assert created["vectors_config"][QdrantReadRepository.DENSE].size == config.vector_size
        assert created["vectors_config"][QdrantReadRepository.DENSE].on_disk
        assert QdrantReadRepository.SPARSE in created["sparse_vectors_config"]
        assert isinstance(created["quantization_config"], ScalarQuantization)
        assert {
            call.kwargs["field_name"]: call.kwargs["field_schema"]
            for call in absent.create_payload_index.call_args_list
        } == collection.payload_indexes

    @staticmethod
    async def test__when_a_collection_exists__returns_no_creation(
        present: AsyncMock, config: QdrantConfig
    ) -> None:
        """Unless it is to be recreated."""
        collection = QdrantCollection(Gauge, present, config)

        await collection.create_collection()

        present.create_collection.assert_not_awaited()
        present.create_payload_index.assert_not_awaited()

    @staticmethod
    async def test__when_a_recreation_is_asked__returns_a_deletion_then_a_creation(
        present: AsyncMock, config: QdrantConfig
    ) -> None:
        """The existing collection is deleted, then created again."""
        present.collection_exists.side_effect = [True, False]
        collection = QdrantCollection(Gauge, present, config)

        await collection.create_collection(recreate_if_exists=True)

        present.delete_collection.assert_awaited_once_with("gauges")
        present.create_collection.assert_awaited_once()

    @staticmethod
    async def test__when_a_collection_exists__returns_a_creation_of_each_index_it_lacks(
        present: AsyncMock, config: QdrantConfig
    ) -> None:
        """One created before a field became filterable."""
        collection = QdrantCollection(Gauge, present, config)

        await collection.ensure_payload_indexes()

        assert present.create_payload_index.await_count == len(collection.payload_indexes)

    @staticmethod
    def test__when_compression_is_binary__returns_in_memory_vectors_and_a_rescore(
        absent: AsyncMock,
    ) -> None:
        """The reader then rescores with its oversampling."""
        config = QdrantConfig(
            top_k=3,
            vector_size=8,
            quantization_type=QuantizationType.BINARY,
            quantization_oversampling=4.0,
        )

        quantization = QdrantCollection(Gauge, absent, config).quantization
        search_params = QdrantReadRepository(Gauge, absent, MagicMock(), config).search_params

        assert isinstance(quantization, BinaryQuantization)
        assert quantization.binary.always_ram
        assert search_params == SearchParams(
            quantization=QuantizationSearchParams(rescore=True, oversampling=4.0)
        )

    @staticmethod
    def test__when_there_is_no_compression__returns_no_quantization_nor_search_params(
        absent: AsyncMock, config: QdrantConfig
    ) -> None:
        """Neither compression nor search parameters."""
        assert QdrantCollection(Gauge, absent, config).quantization is None
        assert QdrantReadRepository(Gauge, absent, MagicMock(), config).search_params is None


class TestConnection:
    """A client is created for the server, its TLS verified against a given certificate."""

    @pytest.fixture
    @staticmethod
    def created(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Record the clients created instead of creating them.

        Returns:
            The stand-in for the client class.
        """
        client_class = MagicMock()
        monkeypatch.setattr(qdrant_collection, "AsyncQdrantClient", client_class)
        monkeypatch.setenv("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", "")
        return client_class

    @staticmethod
    def test__when_a_connection_has_no_certificate__returns_a_client_verifying_as_https_says(
        created: MagicMock,
    ) -> None:
        """No certificate is written; the key is sent in clear to the client."""
        given = SecretStr("key")
        connection = QdrantConnection(host="qdrant", https=True, api_key=given)

        connection.create_client()

        arguments = created.call_args.kwargs
        assert arguments["verify"] is True
        assert arguments["api_key"] == given.get_secret_value()
        assert arguments["host"] == "qdrant"
        assert not os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"]

    @staticmethod
    def test__when_a_certificate_is_given__returns_its_path_for_http_and_grpc(
        created: MagicMock,
    ) -> None:
        """The gRPC channel reads the certificate's path from its environment variable."""
        connection = QdrantConnection(https=True, prefer_grpc=True, tls_cert="PEM")

        connection.create_client()

        path = created.call_args.kwargs["verify"]
        assert Path(path).read_text(encoding="utf-8") == "PEM"
        assert os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] == path
        assert created.call_args.kwargs["prefer_grpc"]
