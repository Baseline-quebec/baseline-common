"""WeaviateCollection — creates and deletes the Weaviate collection of a document type."""

import builtins
import datetime as dt
import types
import typing
from typing import Any, Literal, get_args, get_origin

import weaviate.classes as wvc
from loguru import logger
from weaviate import WeaviateAsyncClient
from weaviate.classes.config import Configure

from baseline_core.core.repository.vectorstore.vector_document import (
    VectorDocument,
    VectorStoreProperty,
)


class WeaviateCollection:
    """The Weaviate collection of one document type: its schema, creation and deletion."""

    def __init__(
        self, content_type: type[VectorDocument], weaviate_client: WeaviateAsyncClient
    ) -> None:
        """Bind the document type and the client.

        Args:
            content_type: The document type; it names the collection and gives its schema.
            weaviate_client: The Weaviate async client.
        """
        self._content_type = content_type
        self._client = weaviate_client
        self._name = content_type.__collection_name__

    @property
    def properties(self) -> list[wvc.config.Property]:
        """The Weaviate properties of the document type's fields, vector fields left out."""
        properties = []

        for field_name, field_info in self._content_type.model_fields.items():
            vectorstore_property = next(
                (
                    metadata
                    for metadata in field_info.metadata
                    if isinstance(metadata, VectorStoreProperty)
                ),
                None,
            )

            # A vector field is embedded on write, not stored as a property.
            if vectorstore_property and vectorstore_property.is_vector:
                continue

            properties.append(
                wvc.config.Property(
                    data_type=self._data_type(field_info.annotation),
                    name=field_name,
                    index_filterable=vectorstore_property.filterable
                    if vectorstore_property
                    else False,
                    index_searchable=vectorstore_property.searchable
                    if vectorstore_property
                    else False,
                )
            )

        return properties

    async def create_collection(self, *, recreate_if_exists: bool = False) -> None:
        """Create the collection, with multi-tenancy, unless it exists.

        Args:
            recreate_if_exists: Whether an existing collection is deleted and created again.
        """
        if recreate_if_exists:
            await self.delete_collection()

        if not await self._client.collections.exists(self._name):
            await self._client.collections.create(
                self._name,
                properties=self.properties,
                multi_tenancy_config=Configure.multi_tenancy(
                    enabled=True, auto_tenant_creation=True
                ),
            )
            logger.info(f"Created collection '{self._name}' with multi-tenancy enabled")

    async def delete_collection(self) -> None:
        """Delete the collection."""
        await self._client.collections.delete(self._name)

    async def schema_has_changed(self) -> bool:
        """Tell whether the collection's properties differ from the document type's fields.

        Returns:
            True when properties were added or removed; False when the collection is missing.
        """
        if not await self._client.collections.exists(self._name):
            return False

        existing_config = await self._client.collections.get(self._name).config.get()
        existing_property_names = {prop.name for prop in existing_config.properties}
        desired_property_names = {prop.name for prop in self.properties}
        return existing_property_names != desired_property_names

    @staticmethod
    def _data_type(annotation: Any) -> wvc.config.DataType:  # noqa: ANN401, PLR0911
        """Map a field's annotation to its Weaviate data type.

        Args:
            annotation: The field's type annotation; ``X | None`` maps as ``X``.

        Returns:
            The Weaviate data type.

        Raises:
            ValueError: When the annotation has no Weaviate data type.
        """
        if get_origin(annotation) in {types.UnionType, typing.Union}:
            annotation = next(arg for arg in get_args(annotation) if arg is not type(None))

        if get_origin(annotation) is Literal:
            return wvc.config.DataType.TEXT

        match annotation:
            case builtins.str:
                return wvc.config.DataType.TEXT
            case builtins.bool:
                return wvc.config.DataType.BOOL
            case builtins.int:
                return wvc.config.DataType.INT
            case builtins.float:
                return wvc.config.DataType.NUMBER
            case dt.datetime:
                return wvc.config.DataType.DATE
            case _ if annotation == list[str]:
                return wvc.config.DataType.TEXT_ARRAY
            case _:
                error_msg = f"Type {annotation} not supported"
                raise ValueError(error_msg)
