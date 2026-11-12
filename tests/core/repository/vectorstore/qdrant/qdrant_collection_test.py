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

from baseline_core.core.repository.vectorstore.qdrant import qdrant_collection
from baseline_core.core.repository.vectorstore.qdrant.qdrant_collection import (
    QdrantCollection,
    QdrantConnection,
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
from tests.core.repository.vectorstore.qdrant.conftest import Machine


class Gauge(VectorDocument):
    """A gauge, with optional and boolean fields."""

    __collection_name__: ClassVar[str] = "gauges"

    calibrated: Annotated[bool, VectorStoreProperty(filterable=True)] = False
    cycles: Annotated[int | None, VectorStoreProperty(filterable=True)] = None
    reading: Annotated[float | None, VectorStoreProperty(filterable=True)] = None
    label: str = ""


class TestPayloadIndexes:
    """Each filterable field gets the index its annotation calls for."""

    @staticmethod
    def test__each_type__gets_its_index(absent: AsyncMock, config: QdrantConfig) -> None:
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
    def test__a_boolean__is_not_indexed_as_a_number(
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
    def test__an_optional_datetime__indexes_by_instant(
        absent: AsyncMock, config: QdrantConfig
    ) -> None:
        """X | None indexes as X."""

        class Stamp(VectorDocument):
            __collection_name__: ClassVar[str] = "stamps"

            at: Annotated[datetime | None, VectorStoreProperty(filterable=True)] = None

        assert QdrantCollection(Stamp, absent, config).payload_indexes == {
            "id_": PayloadSchemaType.KEYWORD,
            "at": PayloadSchemaType.DATETIME,
        }


class TestCreation:
    """A collection is created once, with its vectors, compression and indexes."""

    @staticmethod
    async def test__a_new_collection__gets_its_vectors_and_indexes(absent: AsyncMock) -> None:
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
    async def test__an_existing_collection__is_left_as_it_is(
        present: AsyncMock, config: QdrantConfig
    ) -> None:
        """Unless it is to be recreated."""
        collection = QdrantCollection(Gauge, present, config)

        await collection.create_collection()

        present.create_collection.assert_not_awaited()
        present.create_payload_index.assert_not_awaited()

    @staticmethod
    async def test__a_recreation__deletes_the_collection_first(
        present: AsyncMock, config: QdrantConfig
    ) -> None:
        """The existing collection is deleted, then created again."""
        present.collection_exists.side_effect = [True, False]
        collection = QdrantCollection(Gauge, present, config)

        await collection.create_collection(recreate_if_exists=True)

        present.delete_collection.assert_awaited_once_with("gauges")
        present.create_collection.assert_awaited_once()

    @staticmethod
    async def test__an_existing_collection__gets_the_indexes_it_lacks(
        present: AsyncMock, config: QdrantConfig
    ) -> None:
        """One created before a field became filterable."""
        collection = QdrantCollection(Gauge, present, config)

        await collection.ensure_payload_indexes()

        assert present.create_payload_index.await_count == len(collection.payload_indexes)

    @staticmethod
    def test__binary_compression__keeps_the_vectors_in_memory(absent: AsyncMock) -> None:
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
    def test__no_compression__has_nothing_to_rescore(
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
    def test__a_plain_connection__verifies_as_https_says(created: MagicMock) -> None:
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
    def test__a_certificate__is_verified_against_by_http_and_grpc(created: MagicMock) -> None:
        """The gRPC channel reads the certificate's path from its environment variable."""
        connection = QdrantConnection(https=True, prefer_grpc=True, tls_cert="PEM")

        connection.create_client()

        path = created.call_args.kwargs["verify"]
        assert Path(path).read_text(encoding="utf-8") == "PEM"
        assert os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] == path
        assert created.call_args.kwargs["prefer_grpc"]
