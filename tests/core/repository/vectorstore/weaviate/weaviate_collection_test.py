"""Unit tests for the Weaviate collection: its schema, its creation and deletion."""

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest
import weaviate.classes as wvc
from weaviate.classes.config import Configure

from baseline_common.core.repository.vectorstore.vector_document import VectorDocument
from baseline_common.core.repository.vectorstore.weaviate.weaviate_collection import (
    WeaviateCollection,
)
from tests.core.repository.vectorstore.weaviate.conftest import Machine, WeaviateDouble


class TestProperties:
    """Each field becomes a property of its data type; the vector field is left out."""

    @staticmethod
    def test__each_type__gets_its_data_type(weaviate_double: WeaviateDouble) -> None:
        """An optional field maps as the type it wraps."""
        properties = WeaviateCollection(Machine, weaviate_double.client).properties

        assert {item.name: item.dataType for item in properties} == {
            "id_": wvc.config.DataType.TEXT,
            "name": wvc.config.DataType.TEXT,
            "kind": wvc.config.DataType.TEXT,
            "mode": wvc.config.DataType.TEXT,
            "tags": wvc.config.DataType.TEXT_ARRAY,
            "hours": wvc.config.DataType.NUMBER,
            "shifts": wvc.config.DataType.INT,
            "active": wvc.config.DataType.BOOL,
            "serviced": wvc.config.DataType.DATE,
        }

    @staticmethod
    def test__indexes__follow_the_field_s_properties(weaviate_double: WeaviateDouble) -> None:
        """Searchable and filterable as declared; a field without them is neither."""
        properties = WeaviateCollection(Machine, weaviate_double.client).properties

        indexes = {item.name: (item.indexSearchable, item.indexFilterable) for item in properties}
        assert indexes["name"] == (True, True)
        assert indexes["kind"] == (False, True)
        assert indexes["hours"] == (False, False)

    @staticmethod
    def test__an_unsupported_type__is_refused(weaviate_double: WeaviateDouble) -> None:
        """A field Weaviate cannot store is named when the schema is read."""

        class Gauge(VectorDocument):
            __collection_name__: ClassVar[str] = "gauges"

            readings: dict[str, float]

        with pytest.raises(ValueError, match="not supported"):
            _ = WeaviateCollection(Gauge, weaviate_double.client).properties


class TestCreation:
    """A collection is created once, with multi-tenancy, and recreated on demand."""

    @staticmethod
    async def test__a_new_collection__gets_its_properties_and_tenants(
        weaviate_double: WeaviateDouble,
    ) -> None:
        """Tenants are created on their first write."""
        weaviate_double.client.collections.exists.return_value = False
        collection = WeaviateCollection(Machine, weaviate_double.client)

        await collection.create_collection()

        create = weaviate_double.client.collections.create.await_args
        assert create is not None
        assert create.args == ("machines",)
        assert create.kwargs["properties"] == collection.properties
        assert create.kwargs["multi_tenancy_config"] == Configure.multi_tenancy(
            enabled=True, auto_tenant_creation=True
        )

    @staticmethod
    async def test__an_existing_collection__is_left_as_it_is(
        weaviate_double: WeaviateDouble,
    ) -> None:
        """No create, no delete."""
        await WeaviateCollection(Machine, weaviate_double.client).create_collection()

        weaviate_double.client.collections.create.assert_not_awaited()
        weaviate_double.client.collections.delete.assert_not_awaited()

    @staticmethod
    async def test__a_recreation__deletes_the_collection_first(
        weaviate_double: WeaviateDouble,
    ) -> None:
        """The collection is deleted, then created again."""
        weaviate_double.client.collections.exists.return_value = False

        await WeaviateCollection(Machine, weaviate_double.client).create_collection(
            recreate_if_exists=True
        )

        weaviate_double.client.collections.delete.assert_awaited_once_with("machines")
        weaviate_double.client.collections.create.assert_awaited_once()


class TestSchemaCheck:
    """The collection's properties against the document type's fields."""

    @staticmethod
    def configured(*names: str) -> AsyncMock:
        """Return a config read that answers properties of these names.

        Args:
            *names: The property names Weaviate holds.

        Returns:
            The mocked read.
        """
        properties = [SimpleNamespace(name=name) for name in names]
        return AsyncMock(return_value=SimpleNamespace(properties=properties))

    @staticmethod
    async def test__the_same_properties__have_not_changed(weaviate_double: WeaviateDouble) -> None:
        """Every field is a property, and nothing more."""
        names = [
            item.name for item in WeaviateCollection(Machine, weaviate_double.client).properties
        ]
        weaviate_double.collection.config.get = TestSchemaCheck.configured(*names)

        assert not await WeaviateCollection(Machine, weaviate_double.client).schema_has_changed()

    @staticmethod
    async def test__a_field_added__has_changed(weaviate_double: WeaviateDouble) -> None:
        """The collection lacks a property the document type now has."""
        weaviate_double.collection.config.get = TestSchemaCheck.configured("id_", "name")

        assert await WeaviateCollection(Machine, weaviate_double.client).schema_has_changed()

    @staticmethod
    async def test__a_missing_collection__has_not_changed(weaviate_double: WeaviateDouble) -> None:
        """Nothing to compare: the next creation builds it from the fields."""
        weaviate_double.client.collections.exists.return_value = False

        assert not await WeaviateCollection(Machine, weaviate_double.client).schema_has_changed()


class TestDeletion:
    """A deletion drops the collection by its name."""

    @staticmethod
    async def test__the_collection__is_deleted_by_name(weaviate_double: WeaviateDouble) -> None:
        """The document type names it."""
        await WeaviateCollection(Machine, weaviate_double.client).delete_collection()

        weaviate_double.client.collections.delete.assert_awaited_once_with("machines")
