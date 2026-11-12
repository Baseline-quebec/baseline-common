"""Unit tests for the Qdrant reader: filters, sorting, scrolls, counts and searches."""

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai import Embedder
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import FacetResponse, FacetValueHit, FieldCondition, MatchAny, Range

from baseline_core.core.repository.query_filters import (
    ComparisonFilter,
    EqualityFilter,
    Filter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)
from baseline_core.core.repository.vectorstore.qdrant.qdrant_collection import QdrantCollection
from baseline_core.core.repository.vectorstore.qdrant.qdrant_read_repository import (
    HierarchicalQuery,
    HierarchicalSearchResult,
    LatestQuery,
    OrderNotIndexedError,
    QdrantConfig,
    QdrantReadRepository,
    QuantizationType,
    ValueCount,
    ValuesNotIndexedError,
)
from baseline_core.core.repository.vectorstore.qdrant.qdrant_write_repository import (
    QdrantWriteRepository,
)
from baseline_core.core.repository.vectorstore.vector_document import RetrievedDocument
from tests.core.repository.vectorstore.qdrant.conftest import Machine, Workshop


class MachineName(BaseModel):
    """A projection that keeps the id and the kind."""

    model_config = ConfigDict(extra="allow")

    id_: str
    kind: str


class Serviced(BaseModel):
    """When the latest machine of a workshop was serviced."""

    workshop_id: str
    serviced: datetime


class TestFilters:
    """Every filter becomes a Qdrant condition that keeps the same documents."""

    @staticmethod
    async def test__an_equality__keeps_the_matching_documents(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """The three lathes."""
        found = await stocked.get_by_filters([EqualityFilter(field="kind", value="lathe")])

        assert sorted(machine.id_ for machine in found) == ["m-1", "m-2", "m-5"]

    @staticmethod
    async def test__a_difference__keeps_the_others(stocked: QdrantReadRepository[Machine]) -> None:
        """The press and the mill."""
        found = await stocked.get_by_filters([
            ComparisonFilter(operator="!=", field="kind", value="lathe")
        ])

        assert sorted(machine.id_ for machine in found) == ["m-3", "m-4"]

    @staticmethod
    async def test__equalities_on_one_field__match_any_of_the_values(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """One match on any value, not a disjunction of matches."""
        either = LogicalFilter(
            operator="|",
            operands=[
                EqualityFilter(field="workshop_id", value="w-1"),
                EqualityFilter(field="workshop_id", value="w-3"),
            ],
        )

        qdrant_filter = stocked.qdrant_filter([either])
        found = await stocked.get_by_filters([either])

        assert qdrant_filter is not None
        assert isinstance(qdrant_filter.must, list)
        assert isinstance(qdrant_filter.must[0], FieldCondition)
        assert qdrant_filter.must[0].match == MatchAny(any=["w-1", "w-3"])
        assert sorted(machine.id_ for machine in found) == ["m-1", "m-2", "m-5"]

    @staticmethod
    async def test__either_of_two_fields__keeps_both(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """The press, and the machine run on two shifts."""
        either = LogicalFilter(
            operator="|",
            operands=[
                EqualityFilter(field="kind", value="press"),
                EqualityFilter(field="shifts", value=2),
            ],
        )

        found = await stocked.get_by_filters([either])

        assert sorted(machine.id_ for machine in found) == ["m-1", "m-3"]

    @staticmethod
    async def test__a_float_equality__is_a_closed_range(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """Qdrant matches no float exactly; a range from the value to itself does."""
        twelve_and_a_half = EqualityFilter(field="hours", value=12.5)

        qdrant_filter = stocked.qdrant_filter([twelve_and_a_half])
        found = await stocked.get_by_filters([twelve_and_a_half])

        assert qdrant_filter is not None
        assert isinstance(qdrant_filter.must, list)
        assert isinstance(qdrant_filter.must[0], FieldCondition)
        assert qdrant_filter.must[0].range == Range(gte=12.5, lte=12.5)
        assert sorted(machine.id_ for machine in found) == ["m-1", "m-3"]

    @staticmethod
    async def test__a_number_range__keeps_the_documents_within(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """The machines run on two or three shifts."""
        found = await stocked.get_by_filters([
            ComparisonFilter(operator=">=", field="shifts", value=2),
            ComparisonFilter(operator="<", field="shifts", value=4),
        ])

        assert sorted(machine.id_ for machine in found) == ["m-1", "m-3"]

    @staticmethod
    async def test__a_range_of_iso_text__compares_datetimes(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """The machines serviced in the last ten days of September."""
        found = await stocked.get_by_filters([
            ComparisonFilter(operator=">=", field="serviced", value="2026-09-20T00:00:00Z"),
            ComparisonFilter(operator="<", field="serviced", value="2026-10-01T00:00:00Z"),
        ])

        assert sorted(machine.id_ for machine in found) == ["m-2", "m-3"]

    @staticmethod
    async def test__an_equality_on_a_list__matches_any_element(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """Both machines tagged north."""
        found = await stocked.get_by_filters([EqualityFilter(field="tags", value="north")])

        assert sorted(machine.id_ for machine in found) == ["m-1", "m-2"]

    @staticmethod
    async def test__a_substring__is_refused(stocked: QdrantReadRepository[Machine]) -> None:
        """Qdrant matches words, not substrings: a text filter would keep other documents."""
        with pytest.raises(ValueError, match="TextFilter"):
            await stocked.get_by_filters([TextFilter(field="name", value="ath")])


class TestReads:
    """Documents are read by id, by page, and sorted on an indexed field."""

    @staticmethod
    async def test__an_id__reads_its_document(
        stocked: QdrantReadRepository[Machine], fleet: list[Machine]
    ) -> None:
        """The machine as inserted, its service date included."""
        found = await stocked.get_by_id("m-3")

        assert found == fleet[2]

    @staticmethod
    async def test__an_unknown_id__reads_none(stocked: QdrantReadRepository[Machine]) -> None:
        """No machine m-9."""
        assert await stocked.get_by_id("m-9") is None

    @staticmethod
    async def test__no_collection__reads_nothing(
        machine_reads: QdrantReadRepository[Machine],
    ) -> None:
        """Before the first write, every read is empty."""
        assert await machine_reads.get_by_id("m-1") is None
        assert await machine_reads.get_by_filters() == []
        assert await machine_reads.count_by_filters([]) == 0
        assert await machine_reads.hybrid_search("lathe", "lathe") == []
        assert await machine_reads.count_values("kind") == []

    @staticmethod
    async def test__a_sort__orders_the_pages(stocked: QdrantReadRepository[Machine]) -> None:
        """By shifts: the offset skips the fewest, the descending order starts at the most."""
        ascending = await stocked.get_by_filters(sort_by="shifts", limit=2, offset=1)
        descending = await stocked.get_by_filters(
            sort_by="shifts", sort_order=SortOrder.DESC, limit=2
        )

        assert [machine.id_ for machine in ascending] == ["m-2", "m-1"]
        assert [machine.id_ for machine in descending] == ["m-4", "m-3"]

    @staticmethod
    async def test__no_limit__reads_every_match_past_the_offset(
        stocked: QdrantReadRepository[Machine], fleet: list[Machine]
    ) -> None:
        """Five machines; three skipped leave two."""
        skipped = 3

        every = await stocked.get_by_filters(limit=None)
        rest = await stocked.get_by_filters(limit=None, offset=skipped)

        assert len(every) == len(fleet)
        assert len(rest) == len(fleet) - skipped

    @staticmethod
    async def test__a_sort_without_range_index__is_refused(embedder: Embedder) -> None:
        """The server refuses to sort on a field no range index covers."""
        client = AsyncMock(spec=AsyncQdrantClient)
        client.collection_exists.return_value = True
        client.query_points.side_effect = UnexpectedResponse(
            400,
            "Bad Request",
            b'{"status":{"error":"No range index for `order_by` key: name"}}',
            httpx.Headers(),
        )
        reads = QdrantReadRepository(
            Machine, client, embedder, QdrantConfig(top_k=3, vector_size=8)
        )

        with pytest.raises(OrderNotIndexedError, match="name"):
            await reads.get_by_filters(sort_by="name")

    @staticmethod
    async def test__a_count__counts_every_match(stocked: QdrantReadRepository[Machine]) -> None:
        """Three lathes."""
        lathes: list[Filter] = [EqualityFilter(field="kind", value="lathe")]

        count = await stocked.count_by_filters(lathes)

        assert count == len(await stocked.get_by_filters(lathes))


class TestScroll:
    """A scroll reads a page of the fields a projection declares, and the next page's cursor."""

    @staticmethod
    async def test__a_projection__reads_only_its_fields(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """extra="allow" would keep any other field the store sent back."""
        page = await stocked.scroll(MachineName, [EqualityFilter(field="id_", value="m-3")])

        assert page.rows == [MachineName(id_="m-3", kind="press")]
        assert page.rows[0].model_extra == {}

    @staticmethod
    async def test__the_cursor__pages_through_every_match(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """Three pages of two."""
        seen: list[str] = []
        cursor: str | None = None

        while True:
            page = await stocked.scroll(Machine, limit=2, cursor=cursor)
            seen.extend(machine.id_ for machine in page.rows)
            cursor = page.cursor

            if cursor is None:
                break

        assert sorted(seen) == ["m-1", "m-2", "m-3", "m-4", "m-5"]

    @staticmethod
    async def test__a_payload_that_does_not_fit__fails_loudly(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """A row is never dropped in silence."""

        class Numbered(BaseModel):
            id_: int

        with pytest.raises(ValidationError):
            await stocked.scroll(Numbered)

    @staticmethod
    async def test__an_alias__is_read_by_its_field_name(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """Payloads are written by field name; an alias is not the stored key."""

        class Aliased(BaseModel):
            id_: str
            kind: str = Field(alias="type")

        page = await stocked.scroll(Aliased, [EqualityFilter(field="id_", value="m-3")])

        assert [row.kind for row in page.rows] == ["press"]

    @staticmethod
    async def test__no_collection__scrolls_an_empty_page(
        machine_reads: QdrantReadRepository[Machine],
    ) -> None:
        """No rows, and no next page."""
        page = await machine_reads.scroll(Machine)

        assert page.rows == []
        assert page.cursor is None


class TestCountValues:
    """Each value of a field is counted exactly, most common first."""

    @staticmethod
    async def test__every_value__is_counted(stocked: QdrantReadRepository[Machine]) -> None:
        """Three lathes, then a press and a mill in no set order."""
        counts = await stocked.count_values("kind")

        assert counts[0] == ValueCount(value="lathe", count=3)
        assert sorted(counts[1:], key=lambda count: count.value) == [
            ValueCount(value="mill", count=1),
            ValueCount(value="press", count=1),
        ]

    @staticmethod
    async def test__filters__restrict_the_documents_counted(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """The machines of the second workshop."""
        counts = await stocked.count_values(
            "kind", [EqualityFilter(field="workshop_id", value="w-2")]
        )

        assert sorted(counts, key=lambda count: count.value) == [
            ValueCount(value="mill", count=1),
            ValueCount(value="press", count=1),
        ]

    @staticmethod
    async def test__a_list_field__counts_each_element(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """Two machines tagged north, two tagged south."""
        counts = await stocked.count_values("tags")

        assert sorted(counts, key=lambda count: count.value) == [
            ValueCount(value="north", count=2),
            ValueCount(value="south", count=2),
        ]

    @staticmethod
    async def test__the_limit__stops_at_the_most_common(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """Only the lathes."""
        counts = await stocked.count_values("kind", limit=1)

        assert counts == [ValueCount(value="lathe", count=3)]

    @staticmethod
    async def test__a_value_no_document_holds__is_left_out(embedder: Embedder) -> None:
        """The server's payload index keeps a deleted value, counted 0."""
        client = AsyncMock(spec=AsyncQdrantClient)
        client.collection_exists.return_value = True
        client.facet.return_value = FacetResponse(
            hits=[FacetValueHit(value="lathe", count=3), FacetValueHit(value="gone", count=0)]
        )
        reads = QdrantReadRepository(
            Machine, client, embedder, QdrantConfig(top_k=3, vector_size=8)
        )

        counts = await reads.count_values("kind")

        assert counts == [ValueCount(value="lathe", count=3)]

    @staticmethod
    async def test__a_field_without_index__is_refused(embedder: Embedder) -> None:
        """The server refuses a facet on a field no keyword index covers."""
        client = AsyncMock(spec=AsyncQdrantClient)
        client.collection_exists.return_value = True
        client.facet.side_effect = UnexpectedResponse(
            400,
            "Bad Request",
            b'{"status":{"error":"Wrong input: No appropriate index for faceting: notes"}}',
            httpx.Headers(),
        )
        reads = QdrantReadRepository(
            Machine, client, embedder, QdrantConfig(top_k=3, vector_size=8)
        )

        with pytest.raises(ValuesNotIndexedError, match="notes"):
            await reads.count_values("notes")


class TestLatestPerValue:
    """Each value of a field gets the matching document with the greatest order."""

    @staticmethod
    async def test__each_workshop__gets_its_latest_service(
        stocked: QdrantReadRepository[Machine],
    ) -> None:
        """A machine without a service date is not a candidate."""
        query = LatestQuery(
            group_by="workshop_id",
            order_by="serviced",
            filters=[
                LogicalFilter(
                    operator="|",
                    operands=[
                        EqualityFilter(field="workshop_id", value="w-1"),
                        EqualityFilter(field="workshop_id", value="w-2"),
                    ],
                )
            ],
        )

        latest = await stocked.latest_per_value(Serviced, query)

        assert {row.workshop_id: row.serviced for row in latest} == {
            "w-1": datetime(2026, 9, 20, tzinfo=UTC),
            "w-2": datetime(2026, 9, 30, 10, tzinfo=UTC),
        }


class TestHybridSearch:
    """A search fuses a dense and a sparse ranking, each cut by its own floor."""

    @staticmethod
    async def searched(embedder: Embedder, config: QdrantConfig) -> dict[str, Any]:
        """Search with a mocked client and return what was asked of it.

        Args:
            embedder: The embedder.
            config: The collection config.

        Returns:
            The arguments of the query.
        """
        client = AsyncMock(spec=AsyncQdrantClient)
        client.collection_exists.return_value = True
        client.query_points.return_value = AsyncMock(points=[])
        reads = QdrantReadRepository(Machine, client, embedder, config)

        await reads.hybrid_search("worn spindle", "spindle")

        return cast("dict[str, Any]", client.query_points.call_args.kwargs)

    @staticmethod
    async def test__the_top_k__ranks_the_matches(stocked: QdrantReadRepository[Machine]) -> None:
        """Three lathes at most, ranked from 0, among the lathes."""
        found = await stocked.hybrid_search(
            "lathe", "lathe", [EqualityFilter(field="kind", value="lathe")]
        )

        assert [retrieved.rank for retrieved in found] == [0, 1, 2]
        assert {retrieved.content.id_ for retrieved in found} == {"m-1", "m-2", "m-5"}

    @staticmethod
    async def test__the_dense_floor__stays_off_the_sparse_branch(embedder: Embedder) -> None:
        """Cosine is bounded, BM25 is not: the dense floor would filter nothing there."""
        config = QdrantConfig(top_k=3, vector_size=8, score_threshold=0.25)

        query = await TestHybridSearch.searched(embedder, config)

        assert [prefetch.score_threshold for prefetch in query["prefetch"]] == [0.25, None]

    @staticmethod
    async def test__each_branch__takes_its_own_floor(embedder: Embedder) -> None:
        """A collection that calibrated BM25 filters both branches."""
        config = QdrantConfig(
            top_k=3, vector_size=8, score_threshold=0.25, sparse_score_threshold=3.0
        )

        query = await TestHybridSearch.searched(embedder, config)

        assert [prefetch.score_threshold for prefetch in query["prefetch"]] == [0.25, 3.0]

    @staticmethod
    async def test__the_fused_query__is_never_cut(embedder: Embedder) -> None:
        """RRF scores are rank sums: a floor on them would cut by position."""
        config = QdrantConfig(top_k=3, vector_size=8, score_threshold=0.25)

        query = await TestHybridSearch.searched(embedder, config)

        assert query.get("score_threshold") is None

    @staticmethod
    async def test__the_dense_branch__rescores_like_the_fused_query(embedder: Embedder) -> None:
        """The floor sees the full-precision scores it was calibrated on, not compressed ones."""
        config = QdrantConfig(
            top_k=3, vector_size=8, score_threshold=0.25, quantization_type=QuantizationType.SCALAR
        )

        query = await TestHybridSearch.searched(embedder, config)

        assert query["prefetch"][0].params is not None
        assert query["prefetch"][0].params == query["search_params"]


class TestHierarchicalSearch:
    """Parents are searched first, then the children within those found."""

    @pytest.fixture
    @staticmethod
    async def workshops(
        qdrant_client: AsyncQdrantClient, embedder: Embedder, config: QdrantConfig
    ) -> QdrantReadRepository[Workshop]:
        """Return the reader of three workshops, stored beside the machines.

        Returns:
            The reader.
        """
        reads = QdrantReadRepository(Workshop, qdrant_client, embedder, config)
        writes = QdrantWriteRepository(reads, QdrantCollection(Workshop, qdrant_client, config))
        await writes.insert_many([
            Workshop(id_="w-1", name="North hall"),
            Workshop(id_="w-2", name="South hall"),
            Workshop(id_="w-3", name="Annex"),
        ])
        return reads

    @staticmethod
    async def test__the_parents_found__bound_the_children(
        workshops: QdrantReadRepository[Workshop], stocked: QdrantReadRepository[Machine]
    ) -> None:
        """The north hall, and its two lathes."""
        query = HierarchicalQuery(
            semantic_content="lathe",
            keyword_content="lathe",
            parent_field="id_",
            child_field="workshop_id",
            filters=[EqualityFilter(field="name", value="North hall")],
        )

        result = await workshops.hierarchical_search(query, stocked)

        assert [parent.content.id_ for parent in result.parents] == ["w-1"]
        assert sorted(child.content.id_ for child in result.children) == ["m-1", "m-2"]

    @staticmethod
    async def test__no_parent_found__searches_the_children_alone(
        workshops: QdrantReadRepository[Workshop], stocked: QdrantReadRepository[Machine]
    ) -> None:
        """No workshop has a kind; the press has, and the filter then applies to it."""
        query = HierarchicalQuery(
            semantic_content="press",
            keyword_content="press",
            parent_field="id_",
            child_field="workshop_id",
            filters=[EqualityFilter(field="kind", value="press")],
        )

        result = await workshops.hierarchical_search(query, stocked)

        assert result.parents == []
        assert [child.content.id_ for child in result.children] == ["m-3"]

    @staticmethod
    def test__the_prompt__gives_each_level_its_section(fleet: list[Machine]) -> None:
        """A level without results has no section."""
        result = HierarchicalSearchResult(
            children=[RetrievedDocument(content=fleet[0], score=0.5, rank=0)]
        )

        assert result.has_results
        assert result.prompt.startswith('<children>\n<document rank="1"')
        assert "Machine: Lathe A" in result.prompt
        assert "<parents>" not in result.prompt
        assert not HierarchicalSearchResult().has_results

    @staticmethod
    def test__a_limit_below_one__is_refused() -> None:
        """A limit of 0 would empty the result."""
        with pytest.raises(ValidationError):
            HierarchicalQuery(
                semantic_content="",
                keyword_content="",
                parent_field="id_",
                child_field="workshop_id",
                parent_limit=0,
            )
