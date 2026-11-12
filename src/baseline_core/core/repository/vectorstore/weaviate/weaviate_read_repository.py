"""WeaviateReadRepository — reads documents from a Weaviate collection, by filters or meaning."""

from collections.abc import Sequence
from typing import Any, ClassVar, NewType

import weaviate
import weaviate.classes.query as wcq
from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Embedder
from weaviate import WeaviateAsyncClient
from weaviate.classes.query import HybridFusion, MetadataQuery, Sort

from baseline_core.core.repository.abstract_repository import AbstractReadRepository
from baseline_core.core.repository.query_filters import (
    ComparisonFilter,
    Filter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)
from baseline_core.core.repository.vectorstore.vector_document import (
    RetrievedDocument,
    VectorDocument,
)


# A type of its own, so a tenant can be injected by type.
TenantID = NewType("TenantID", str)


class WeaviateConfig(BaseModel):
    """How a Weaviate collection is searched, and for which tenant."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    top_k: int
    """The number of results to return."""

    hybrid_weight: float
    """The value used to weight the semantic and keyword matching scores for the retrieval."""

    tenant: TenantID
    """The tenant the documents belong to; empty for a collection without tenants."""

    fusion_type: HybridFusion = HybridFusion.RELATIVE_SCORE
    """How the semantic and keyword scores are fused."""


class WeaviateReadRepository[T: VectorDocument](AbstractReadRepository[T]):
    """Reads one document type from its Weaviate collection, for one tenant."""

    COMPARISONS: ClassVar[dict[str, str]] = {
        "==": "equal",
        "!=": "not_equal",
        ">": "greater_than",
        "<": "less_than",
        ">=": "greater_or_equal",
        "<=": "less_or_equal",
    }

    def __init__(
        self,
        content_type: type[T],
        weaviate_client: WeaviateAsyncClient,
        embedding_model: Embedder,
        config: WeaviateConfig,
    ) -> None:
        """Bind the collection of the document type, for the configured tenant.

        Args:
            content_type: The document type; it names the collection.
            weaviate_client: The Weaviate async client.
            embedding_model: Embeds the semantic content of a search.
            config: How the collection is searched, and for which tenant.
        """
        super().__init__(content_type)
        self.weaviate_client = weaviate_client
        self.embedding_model = embedding_model
        self.config = config
        self.collection = weaviate_client.collections.get(content_type.__collection_name__)

        if config.tenant:
            self.collection = self.collection.with_tenant(config.tenant)

    async def hybrid_search(
        self,
        semantic_content: str,
        keyword_content: str,
        filters: list[Filter] | None = None,
        limit: int | None = None,
        hybrid_weight: float | None = None,
    ) -> list[RetrievedDocument]:
        """Search the collection by meaning and by keywords at once.

        Args:
            semantic_content: The semantic content of the query.
            keyword_content: The keyword content of the query.
            filters: The conditions, all of which must hold.
            limit: The most documents returned; the configured top_k by default.
            hybrid_weight: The weight of the semantic score; the configured one by default.

        Returns:
            The documents retrieved, ranked; none when the collection does not exist.

        Raises:
            ValueError: When Weaviate returns a document without its score.
        """
        if not await self.weaviate_client.collections.exists(
            self.content_type.__collection_name__
        ):
            return []

        content_to_embed = semantic_content if len(semantic_content) > 0 else keyword_content
        embedding_result = await self.embedding_model.embed_query([
            self.content_type.__embedding_prefix__ + content_to_embed
        ])
        vector = list(embedding_result.embeddings[0])

        objects = await self.collection.query.hybrid(
            vector=vector,
            query=keyword_content,
            filters=self._where(filters or []),
            limit=self.config.top_k if limit is None else limit,
            alpha=self.config.hybrid_weight if hybrid_weight is None else hybrid_weight,
            fusion_type=self.config.fusion_type,
            return_metadata=MetadataQuery(score=True),
        )
        documents_retrieved = []

        for i, object_ in enumerate(objects.objects):
            if object_.metadata.score is None:
                error_msg = f"Weaviate returned no score for {object_.properties['id_']}"
                raise ValueError(error_msg)

            document = self.content_type.model_validate(object_.properties)
            documents_retrieved.append(
                RetrievedDocument(content=document, score=object_.metadata.score, rank=i)
            )

        return documents_retrieved

    async def get_by_filters(
        self,
        filters: Sequence[Filter] = (),
        sort_by: str | None = None,
        sort_order: SortOrder = SortOrder.ASC,
        limit: int | None = 100,
        offset: int = 0,
        *,
        include_vector: bool = True,
    ) -> list[T]:
        """Find the documents every filter holds for.

        Args:
            filters: The conditions, all of which must hold.
            sort_by: The property to sort on.
            sort_order: The order to sort in when sort_by is given.
            limit: The most documents returned.
            offset: How many matching documents to skip.
            include_vector: Whether each document carries its vector.

        Returns:
            The matching documents.
        """
        objects = await self.collection.query.fetch_objects(
            filters=self._where(filters),
            limit=limit,
            offset=offset,
            sort=Sort.by_property(sort_by, sort_order == SortOrder.ASC) if sort_by else None,
            include_vector=include_vector,
        )
        vector_name = self.content_type.__vector_field_name__
        documents_retrieved = []

        for object_ in objects.objects:
            # Weaviate nests the vector under its name; the model holds it in its own field.
            vector = (
                {vector_name: object_.vector.get("default")}
                if object_.vector and include_vector
                else {}
            )
            properties = {**object_.properties, **vector}  # type: ignore[misc]
            documents_retrieved.append(self.content_type.model_validate(properties))

        return documents_retrieved

    async def get_by_id(self, id_: str) -> T | None:
        """Find a document by its id.

        Args:
            id_: The document's id_ value.

        Returns:
            The document, or None when there is none or the tenant does not exist yet.
        """
        try:
            response = await self.collection.query.fetch_objects(
                filters=wcq.Filter.by_property("id_").equal(id_), limit=1
            )
        except weaviate.exceptions.WeaviateQueryError as err:
            if "tenant not found" in str(err):
                logger.warning("Document with id {} not found (tenant not initialized)", id_)
                return None

            raise

        if not response.objects:
            logger.info("Document with id {} not found", id_)
            return None

        return self.content_type.model_validate(response.objects[0].properties)

    def _where(self, filters: Sequence[Filter]) -> Any:  # noqa: ANN401
        """Combine conditions into the Weaviate filter all of them must hold for.

        Args:
            filters: The conditions.

        Returns:
            The Weaviate filter, or None when there is no condition.
        """
        if not filters:
            return None

        return wcq.Filter.all_of([self._weaviate_filter(condition) for condition in filters])

    def _weaviate_filter(self, condition: Filter) -> Any:  # noqa: ANN401
        """Translate a condition into a Weaviate filter.

        Args:
            condition: The condition; a text filter matches its value anywhere in the text.

        Returns:
            The Weaviate filter.

        Raises:
            ValueError: When the condition's operator has no Weaviate counterpart.
        """
        if isinstance(condition, LogicalFilter):
            operands = [self._weaviate_filter(operand) for operand in condition.operands]
            combine = wcq.Filter.all_of if condition.operator == "&" else wcq.Filter.any_of
            return combine(operands)

        if isinstance(condition, TextFilter):
            return wcq.Filter.by_property(condition.field).like(f"*{condition.value}*")

        if isinstance(condition, ComparisonFilter):
            compare = getattr(
                wcq.Filter.by_property(condition.field), self.COMPARISONS[condition.operator]
            )
            return compare(condition.value)

        error_msg = f"Operator {condition.operator} not supported"
        raise ValueError(error_msg)
