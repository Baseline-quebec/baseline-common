"""QdrantReadRepository — reads documents from a Qdrant collection, by filters or meaning."""

import uuid
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Embedder
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    DatetimeRange,
    Direction,
    Distance,
    Document,
    FieldCondition,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    OrderBy,
    OrderByQuery,
    Prefetch,
    QuantizationSearchParams,
    Range,
    SearchParams,
)
from qdrant_client.models import Filter as QdrantFilter

from baseline_common.core.repository.abstract_repository import AbstractReadRepository
from baseline_common.core.repository.query_filters import (
    ComparisonFilter,
    EqualityFilter,
    Filter,
    LogicalFilter,
    SortOrder,
)
from baseline_common.core.repository.vectorstore.vector_document import (
    RetrievedDocument,
    VectorDocument,
)


class QuantizationType(StrEnum):
    """How dense vectors are compressed: not at all, to bytes (~4x), or to bits (~32x)."""

    NONE = "none"
    SCALAR = "scalar"
    BINARY = "binary"


class QdrantConfig(BaseModel):
    """How a Qdrant collection is built and searched."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    top_k: int
    """The number of results a search returns by default."""

    vector_size: int
    """The size of the dense vectors, that of the embedding model."""

    distance_metric: Distance = Distance.COSINE
    """The distance dense vectors are compared by."""

    use_sparse_vectors: bool = True
    """Whether a sparse (BM25) vector is stored and searched beside the dense one."""

    sparse_model_name: str = "Qdrant/bm25"
    """The FastEmbed model that turns text into a sparse vector."""

    on_disk: bool = False
    """Whether vectors are kept on disk rather than in memory."""

    quantization_type: QuantizationType = QuantizationType.NONE
    """How dense vectors are compressed."""

    quantization_always_ram: bool = True
    """Whether compressed vectors stay in memory."""

    quantization_rescore: bool = True
    """Whether candidates found on compressed vectors are rescored on the original ones."""

    quantization_oversampling: float = 2.0
    """How many times more candidates are rescored: about 2 for scalar, 3 to 5 for binary."""

    score_threshold: float | None = None
    """The similarity below which a dense candidate is dropped; None keeps every one."""

    sparse_score_threshold: float | None = None
    """The BM25 score below which a sparse candidate is dropped, calibrated on its corpus."""

    batch_size: int = Field(default=50, gt=0)
    """How many documents a write embeds and sends at once."""


class ValueCount(BaseModel):
    """How many documents hold one value of a field."""

    value: str
    count: int


class ScrollPage[P: BaseModel](BaseModel):
    """One page of a scroll, and the cursor to the next."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    rows: list[P]
    """The documents of the page, read as the projection."""

    cursor: str | None
    """The cursor to the next page; None once every match is read."""


class LatestQuery(BaseModel):
    """Which documents a latest-per-value read groups, and what decides the latest."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    group_by: str
    """The field whose values form the groups."""

    order_by: str
    """The field whose greatest value wins; documents without it are left out."""

    filters: list[Filter] = []
    """The conditions a grouped document meets."""

    limit: int = 100
    """The most groups returned."""


class HierarchicalQuery(BaseModel):
    """A search of parents, such as workshops, then of their children, such as machines."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    semantic_content: str
    """The text searched by meaning."""

    keyword_content: str
    """The text searched by keywords."""

    parent_field: str
    """The parent field whose value a child holds, such as ``id_``."""

    child_field: str
    """The child field holding its parent's value, such as ``workshop_id``."""

    filters: list[Filter] = []
    """The conditions on the parents; on the children too when no parent matches."""

    parent_limit: int = Field(5, ge=1)
    """The most parents returned."""

    child_limit: int = Field(10, ge=1)
    """The most children returned."""


class HierarchicalSearchResult(BaseModel):
    """The parents a search found, and the children found within them."""

    parents: list[RetrievedDocument] = []
    children: list[RetrievedDocument] = []

    @property
    def has_results(self) -> bool:
        """Whether anything was found, at either level."""
        return bool(self.parents or self.children)

    @property
    def prompt(self) -> str:
        """The documents found, each level in its own section, for an LLM to read."""
        levels = {"parents": self.parents, "children": self.children}
        sections = [
            f"<{level}>\n{'\n'.join(found.prompt for found in retrieved)}\n</{level}>"
            for level, retrieved in levels.items()
            if retrieved
        ]
        return "\n\n".join(sections)


class ValuesNotIndexedError(LookupError):
    """A field's values cannot be counted: no payload index covers it."""


class OrderNotIndexedError(LookupError):
    """Documents cannot be sorted on a field: no range index covers it."""


class QdrantReadRepository[T: VectorDocument](AbstractReadRepository[T]):
    """Reads one document type from its Qdrant collection, by filters or by meaning."""

    DENSE: ClassVar[str] = "dense"
    SPARSE: ClassVar[str] = "sparse"
    BOUNDS: ClassVar[dict[str, str]] = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte"}

    def __init__(
        self,
        content_type: type[T],
        qdrant_client: AsyncQdrantClient,
        embedding_model: Embedder,
        config: QdrantConfig,
    ) -> None:
        """Bind the collection of the document type.

        Args:
            content_type: The document type; it names the collection.
            qdrant_client: The Qdrant async client.
            embedding_model: Embeds the semantic content of a search.
            config: How the collection is built and searched.
        """
        super().__init__(content_type)
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.config = config
        self.collection_name = content_type.__collection_name__

    @property
    def search_params(self) -> SearchParams | None:
        """How a search rescores candidates found on compressed vectors; None without them."""
        if self.config.quantization_type == QuantizationType.NONE:
            return None

        return SearchParams(
            quantization=QuantizationSearchParams(
                rescore=self.config.quantization_rescore,
                oversampling=self.config.quantization_oversampling,
            )
        )

    async def hybrid_search(
        self,
        semantic_content: str,
        keyword_content: str,
        filters: list[Filter] | None = None,
        limit: int | None = None,
    ) -> list[RetrievedDocument]:
        """Search by meaning and by keywords at once, the two rankings fused by RRF.

        Args:
            semantic_content: The text searched by meaning; the keywords when empty.
            keyword_content: The text searched by keywords on the sparse vectors.
            filters: The conditions, all of which must hold.
            limit: The most documents returned; the configured top_k by default.

        Returns:
            The documents retrieved, ranked.
        """
        if not await self.qdrant_client.collection_exists(self.collection_name):
            return []

        search_limit = limit or self.config.top_k
        prefetch_limit = search_limit * 2
        query_text = semantic_content or keyword_content
        embedding = await self.embedding_model.embed_query([
            self.content_type.__embedding_prefix__ + query_text
        ])

        # The floors cut each branch, not the fused result: RRF scores are rank sums, so a floor
        # on them would cut by position. Cosine and BM25 scales differ, so each has its own.
        prefetch = [
            Prefetch(
                query=list(embedding.embeddings[0]),
                using=self.DENSE,
                limit=prefetch_limit,
                score_threshold=self.config.score_threshold,
                params=self.search_params,
            )
        ]

        if self.config.use_sparse_vectors and keyword_content:
            prefetch.append(
                Prefetch(
                    query=Document(text=keyword_content, model=self.config.sparse_model_name),
                    using=self.SPARSE,
                    limit=prefetch_limit,
                    score_threshold=self.config.sparse_score_threshold,
                )
            )

        response = await self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=self.qdrant_filter(filters or []),
            limit=search_limit,
            with_payload=True,
            search_params=self.search_params,
        )
        return [
            RetrievedDocument(
                content=self.validated(self.content_type, point.payload),
                score=point.score,
                rank=rank,
            )
            for rank, point in enumerate(response.points)
        ]

    async def hierarchical_search(
        self, query: HierarchicalQuery, children: "QdrantReadRepository[Any]"
    ) -> HierarchicalSearchResult:
        """Search the parents, then the children of those found; the children alone if none.

        Args:
            query: What is searched, and how a child names its parent.
            children: Reads the children's collection.

        Returns:
            The parents found and the children found within them.
        """
        parents = await self.hybrid_search(
            query.semantic_content, query.keyword_content, query.filters, query.parent_limit
        )

        if not parents:
            found = await children.hybrid_search(
                query.semantic_content, query.keyword_content, query.filters, query.child_limit
            )
            return HierarchicalSearchResult(children=found)

        keys = dict.fromkeys(getattr(parent.content, query.parent_field) for parent in parents)
        within: list[Filter] = [
            LogicalFilter(
                operator="|",
                operands=[EqualityFilter(field=query.child_field, value=key) for key in keys],
            )
        ]
        found = await children.hybrid_search(
            query.semantic_content, query.keyword_content, within, query.child_limit
        )
        return HierarchicalSearchResult(parents=parents, children=found)

    async def get_by_id(self, id_: str) -> T | None:
        """Find a document by its id.

        Args:
            id_: The document's id_ value.

        Returns:
            The document, or None when there is none or no collection yet.
        """
        if not await self.qdrant_client.collection_exists(self.collection_name):
            return None

        points = await self.qdrant_client.retrieve(
            collection_name=self.collection_name, ids=[self.point_id(id_)], with_payload=True
        )
        return self.validated(self.content_type, points[0].payload) if points else None

    async def get_by_filters(
        self,
        filters: Sequence[Filter] = (),
        sort_by: str | None = None,
        sort_order: SortOrder = SortOrder.ASC,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[T]:
        """Find the documents every filter holds for.

        Args:
            filters: The conditions, all of which must hold.
            sort_by: The field to sort on; it needs a range index (a number or a datetime).
            sort_order: The order to sort in when sort_by is given.
            limit: The most documents returned; None returns them all.
            offset: How many matching documents to skip.

        Returns:
            The matching documents.

        Raises:
            OrderNotIndexedError: When no range index covers sort_by.
        """
        if not await self.qdrant_client.collection_exists(self.collection_name):
            return []

        qdrant_filter = self.qdrant_filter(filters)

        if limit is None:
            limit = await self.count_by_filters(list(filters)) - offset

        if limit <= 0:
            return []

        order = (
            OrderByQuery(order_by=OrderBy(key=sort_by, direction=Direction(sort_order.value)))
            if sort_by
            else None
        )

        try:
            response = await self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=order,
                query_filter=qdrant_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
            )
        except UnexpectedResponse as error:
            # Sorting needs a range index; a keyword one, or none, is refused.
            if b"No range index" in error.content:
                msg = f"No range index covers {sort_by!r}: documents cannot be sorted on it"
                raise OrderNotIndexedError(msg) from error

            raise

        return [self.validated(self.content_type, point.payload) for point in response.points]

    async def scroll[P: BaseModel](
        self,
        projection: type[P],
        filters: list[Filter] | None = None,
        limit: int = 1000,
        cursor: str | None = None,
    ) -> ScrollPage[P]:
        """Read a page of the matching documents, only the fields a projection declares.

        Args:
            projection: The model whose fields name what is read; the document type reads all.
            filters: The conditions, all of which must hold.
            limit: The most documents in the page.
            cursor: The cursor a previous page gave; None starts from the beginning.

        Returns:
            The page, and the cursor to the next one.
        """
        if not await self.qdrant_client.collection_exists(self.collection_name):
            return ScrollPage[P](rows=[], cursor=None)

        points, next_offset = await self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self.qdrant_filter(filters or []),
            limit=limit,
            offset=cursor,
            with_payload=list(projection.model_fields),
            with_vectors=False,
        )
        return ScrollPage[P](
            rows=[self.validated(projection, point.payload) for point in points],
            cursor=None if next_offset is None else str(next_offset),
        )

    async def count_by_filters(self, filters: list[Filter]) -> int:
        """Count the documents every filter holds for, exactly.

        Args:
            filters: The conditions, all of which must hold.

        Returns:
            The number of matching documents.
        """
        if not await self.qdrant_client.collection_exists(self.collection_name):
            return 0

        result = await self.qdrant_client.count(
            collection_name=self.collection_name,
            count_filter=self.qdrant_filter(filters),
            exact=True,
        )
        return int(result.count)

    async def count_values(
        self, field: str, filters: list[Filter] | None = None, limit: int = 100
    ) -> list[ValueCount]:
        """Count the documents holding each value of a field, exactly, most common first.

        Args:
            field: The field; a list field counts each of its elements.
            filters: The conditions a counted document meets.
            limit: The most values returned.

        Returns:
            One count per value; none when there is no collection yet.

        Raises:
            ValuesNotIndexedError: When no payload index covers the field.
        """
        if not await self.qdrant_client.collection_exists(self.collection_name):
            return []

        try:
            result = await self.qdrant_client.facet(
                collection_name=self.collection_name,
                key=field,
                facet_filter=self.qdrant_filter(filters or []),
                limit=limit,
                exact=True,
            )
        except UnexpectedResponse as error:
            # A facet needs a keyword index; the server refuses a field without one.
            if b"No appropriate index" in error.content:
                msg = f"No payload index covers {field!r}: its values cannot be counted"
                raise ValuesNotIndexedError(msg) from error

            raise

        # The payload index keeps a value its documents no longer hold, counted 0.
        return [
            ValueCount(value=str(hit.value), count=hit.count) for hit in result.hits if hit.count
        ]

    async def latest_per_value[P: BaseModel](
        self, projection: type[P], query: LatestQuery
    ) -> list[P]:
        """Read, for each value of a field, the matching document with the greatest order.

        Args:
            projection: The model whose fields name what is read.
            query: Which documents are grouped, on what, and what decides the latest.

        Returns:
            One row per value, in no set order.

        Raises:
            OrderNotIndexedError: When no range index covers the order field.
        """
        if not await self.qdrant_client.collection_exists(self.collection_name):
            return []

        try:
            result = await self.qdrant_client.query_points_groups(
                collection_name=self.collection_name,
                group_by=query.group_by,
                query=OrderByQuery(order_by=OrderBy(key=query.order_by, direction=Direction.DESC)),
                query_filter=self.qdrant_filter(query.filters),
                limit=query.limit,
                group_size=1,
                with_payload=list(projection.model_fields),
            )
        except UnexpectedResponse as error:
            # Sorting needs a range index; a keyword one, or none, is refused.
            if b"No range index" in error.content:
                msg = f"No range index covers {query.order_by!r}: documents cannot be sorted on it"
                raise OrderNotIndexedError(msg) from error

            raise

        return [
            self.validated(projection, group.hits[0].payload)
            for group in result.groups
            if group.hits
        ]

    def qdrant_filter(self, filters: Sequence[Filter]) -> QdrantFilter | None:
        """Combine conditions into the Qdrant filter all of them must hold for.

        Args:
            filters: The conditions.

        Returns:
            The Qdrant filter, or None when there is no condition.
        """
        if not filters:
            return None

        return QdrantFilter(must=[self._condition(condition) for condition in filters])

    @staticmethod
    def point_id(id_: str) -> str:
        """Derive the Qdrant point id of a document, the same for every read and write.

        Args:
            id_: The document's id_ value.

        Returns:
            The point id, a UUID as text.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_))

    @staticmethod
    def validated[P: BaseModel](model: type[P], payload: dict[str, Any] | None) -> P:
        """Read a payload as a model, by field name: payloads are written by name, not alias.

        Args:
            model: The document type, or a projection of it.
            payload: The payload of a point.

        Returns:
            The payload as the model; a payload that does not fit raises, never skipped.
        """
        return model.model_validate(payload or {}, by_name=True, by_alias=False)

    def _condition(self, condition: Filter) -> FieldCondition | QdrantFilter:
        """Translate a condition into a Qdrant condition.

        Args:
            condition: The condition.

        Returns:
            The Qdrant condition.

        Raises:
            ValueError: When Qdrant has no counterpart, as for a substring match.
        """
        if isinstance(condition, LogicalFilter):
            any_of = self._match_any(condition)

            if any_of is not None:
                return any_of

            operands = [self._condition(operand) for operand in condition.operands]
            return (
                QdrantFilter(must=operands)
                if condition.operator == "&"
                else QdrantFilter(should=operands)
            )

        if isinstance(condition, ComparisonFilter):
            return self._comparison(condition)

        error_msg = f"Qdrant cannot apply a {type(condition).__name__} ({condition.operator})"
        raise ValueError(error_msg)

    def _comparison(self, condition: ComparisonFilter) -> FieldCondition | QdrantFilter:
        """Translate a comparison of a field with a value into a Qdrant condition.

        Args:
            condition: The comparison; text compared by order is taken as a datetime.

        Returns:
            The Qdrant condition.
        """
        if condition.operator in {"==", "!="}:
            # MatchValue takes no float; a float equals a value within a closed range.
            equal = (
                FieldCondition(
                    key=condition.field, range=Range(gte=condition.value, lte=condition.value)
                )
                if isinstance(condition.value, float)
                else FieldCondition(key=condition.field, match=MatchValue(value=condition.value))
            )
            return equal if condition.operator == "==" else QdrantFilter(must_not=[equal])

        # A datetime index compares ISO text as instants; Range only compares numbers.
        bound: dict[str, Any] = {self.BOUNDS[condition.operator]: condition.value}
        range_ = DatetimeRange(**bound) if isinstance(condition.value, str) else Range(**bound)
        return FieldCondition(key=condition.field, range=range_)

    @staticmethod
    def _match_any(condition: LogicalFilter) -> FieldCondition | None:
        """Turn an either-or of equalities on one field into one match on any of the values.

        Args:
            condition: The logical condition.

        Returns:
            The match, or None unless all are text, or all integer, equalities on one field.
        """
        equalities = [
            operand
            for operand in condition.operands
            if isinstance(operand, ComparisonFilter) and operand.operator == "=="
        ]

        if (
            condition.operator != "|"
            or not equalities
            or len(equalities) != len(condition.operands)
            or len({equality.field for equality in equalities}) != 1
        ):
            return None

        texts = [equality.value for equality in equalities if isinstance(equality.value, str)]
        integers = [
            equality.value
            for equality in equalities
            if isinstance(equality.value, int) and not isinstance(equality.value, bool)
        ]

        if len(equalities) not in {len(texts), len(integers)}:
            return None

        return FieldCondition(key=equalities[0].field, match=MatchAny(any=texts or integers))
