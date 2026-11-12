"""Unit tests for the Weaviate reader: by id, by filters, and the hybrid search."""

import pytest
import weaviate.classes.query as wcq
from pydantic_ai import Embedder
from weaviate.collections.classes.internal import MetadataReturn, Object, QueryReturn
from weaviate.exceptions import WeaviateQueryError

from baseline_common.core.repository.query_filters import (
    ComparisonFilter,
    EqualityFilter,
    LogicalFilter,
    SortOrder,
    TextFilter,
)
from baseline_common.core.repository.vectorstore.weaviate.weaviate_read_repository import (
    TenantID,
    WeaviateConfig,
    WeaviateReadRepository,
)
from tests.core.repository.vectorstore.weaviate.conftest import Machine, WeaviateDouble


class TestTenant:
    """A tenant scopes every read; an empty one reads the collection itself."""

    @staticmethod
    def test__a_tenant__scopes_the_collection(
        weaviate_double: WeaviateDouble, embedder: Embedder
    ) -> None:
        """The collection is read through its tenant."""
        config = WeaviateConfig(top_k=3, hybrid_weight=0.5, tenant=TenantID("t-1"))

        WeaviateReadRepository(Machine, weaviate_double.client, embedder, config)

        weaviate_double.client.collections.get.assert_called_once_with("machines")
        weaviate_double.collection.with_tenant.assert_called_once_with("t-1")

    @staticmethod
    @pytest.mark.usefixtures("machine_reads")
    def test__no_tenant__reads_the_collection_itself(weaviate_double: WeaviateDouble) -> None:
        """A collection without tenants is never scoped."""
        weaviate_double.collection.with_tenant.assert_not_called()


class TestGetById:
    """A document by its id_, or None."""

    @staticmethod
    async def test__a_document_held__is_read_as_the_model(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """Its properties validate into the document type."""
        weaviate_double.objects["a"] = WeaviateDouble.stored({"id_": "m-1", "name": "Lathe A"})

        found = await machine_reads.get_by_id("m-1")

        assert found == Machine(id_="m-1", name="Lathe A")
        weaviate_double.collection.query.fetch_objects.assert_awaited_once_with(
            filters=wcq.Filter.by_property("id_").equal("m-1"), limit=1
        )

    @staticmethod
    async def test__no_document__is_none(machine_reads: WeaviateReadRepository[Machine]) -> None:
        """An empty answer is no document."""
        assert await machine_reads.get_by_id("m-9") is None

    @staticmethod
    async def test__a_tenant_not_yet_created__is_none(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """A tenant appears on its first write: before it, nothing is held."""
        weaviate_double.collection.query.fetch_objects.side_effect = WeaviateQueryError(
            "tenant not found: t-1", "GRPC search"
        )

        assert await machine_reads.get_by_id("m-1") is None

    @staticmethod
    async def test__any_other_query_error__is_raised(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """Only a missing tenant means no document."""
        weaviate_double.collection.query.fetch_objects.side_effect = WeaviateQueryError(
            "connection reset", "GRPC search"
        )

        with pytest.raises(WeaviateQueryError, match="connection reset"):
            await machine_reads.get_by_id("m-1")


class TestGetByFilters:
    """Filters, sort, limit and offset become one Weaviate fetch."""

    @staticmethod
    def spelled(where: object) -> object:
        """Spell a Weaviate filter out, as its and/or nodes do not compare.

        Args:
            where: The Weaviate filter.

        Returns:
            The filter, each and/or node as its name and its spelled operands.
        """
        operands = getattr(where, "filters", None)

        if operands is None:
            return where

        kind = type(where).__name__.removeprefix("_Filter").lower()
        return [kind, [TestGetByFilters.spelled(operand) for operand in operands]]

    @staticmethod
    async def test__filters__are_all_of_their_weaviate_counterparts(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """An equality, a comparison, a text search and an or, all of which must hold."""
        filters = [
            EqualityFilter(field="kind", value="press"),
            ComparisonFilter(field="hours", operator=">=", value=3),
            TextFilter(field="name", value="Lathe"),
            LogicalFilter(
                operator="|",
                operands=[
                    EqualityFilter(field="mode", value="auto"),
                    ComparisonFilter(field="shifts", operator="!=", value=0),
                ],
            ),
        ]

        await machine_reads.get_by_filters(filters, limit=10, offset=5)

        fetch = weaviate_double.collection.query.fetch_objects.await_args
        assert fetch is not None
        assert TestGetByFilters.spelled(fetch.kwargs["filters"]) == [
            "and",
            [
                wcq.Filter.by_property("kind").equal("press"),
                wcq.Filter.by_property("hours").greater_or_equal(3),
                wcq.Filter.by_property("name").like("*Lathe*"),
                [
                    "or",
                    [
                        wcq.Filter.by_property("mode").equal("auto"),
                        wcq.Filter.by_property("shifts").not_equal(0),
                    ],
                ],
            ],
        ]
        assert (fetch.kwargs["limit"], fetch.kwargs["offset"]) == (10, 5)
        assert fetch.kwargs["sort"] is None

    @staticmethod
    async def test__no_filter__fetches_without_a_where(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """Every document, up to the default limit."""
        await machine_reads.get_by_filters()

        fetch = weaviate_double.collection.query.fetch_objects.await_args
        assert fetch is not None
        assert (fetch.kwargs["filters"], fetch.kwargs["limit"]) == (None, 100)

    @staticmethod
    @pytest.mark.parametrize(
        ("order", "expected"),
        [(SortOrder.ASC, ["hours", True]), (SortOrder.DESC, ["hours", False])],
    )
    async def test__a_sort__is_on_the_property_in_its_order(
        weaviate_double: WeaviateDouble,
        machine_reads: WeaviateReadRepository[Machine],
        order: SortOrder,
        expected: list[object],
    ) -> None:
        """The sort names the property and its direction."""
        await machine_reads.get_by_filters(sort_by="hours", sort_order=order)

        fetch = weaviate_double.collection.query.fetch_objects.await_args
        assert fetch is not None
        (sort,) = fetch.kwargs["sort"].sorts
        assert [sort.prop, sort.ascending] == expected

    @staticmethod
    async def test__the_vector__is_read_into_the_vector_field(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """Weaviate nests it under its name; the document holds it in its own field."""
        weaviate_double.objects["a"] = WeaviateDouble.stored(
            {"id_": "m-1", "name": "Lathe A"}, [0.5, 0.5, 0.5, 0.5]
        )

        (found,) = await machine_reads.get_by_filters()

        assert found.vector == [0.5, 0.5, 0.5, 0.5]

    @staticmethod
    async def test__without_the_vector__the_field_stays_unset(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """A read that leaves vectors out does not ask for them."""
        weaviate_double.objects["a"] = WeaviateDouble.stored(
            {"id_": "m-1", "name": "Lathe A"}, [0.5, 0.5, 0.5, 0.5]
        )

        (found,) = await machine_reads.get_by_filters(include_vector=False)

        assert found.vector is None
        fetch = weaviate_double.collection.query.fetch_objects.await_args
        assert fetch is not None
        assert fetch.kwargs["include_vector"] is False


class TestHybridSearch:
    """A search by meaning and keywords at once, its results ranked by score."""

    @staticmethod
    def scored(id_: str, name: str, score: float | None) -> Object[dict[str, str], None]:
        """Build a search hit.

        Args:
            id_: The document's id_.
            name: Its name.
            score: The score Weaviate gives it.

        Returns:
            The hit.
        """
        hit = WeaviateDouble.stored({"id_": id_, "name": name})
        return Object(
            uuid=hit.uuid,
            metadata=MetadataReturn(score=score),
            properties=hit.properties,
            references=None,
            vector={},
            collection=hit.collection,
        )

    @staticmethod
    async def test__hits__come_back_ranked_with_their_scores(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """Rank follows Weaviate's order; the score is Weaviate's own."""
        weaviate_double.collection.query.hybrid.return_value = QueryReturn(
            objects=[
                TestHybridSearch.scored("m-2", "Press B", 0.9),
                TestHybridSearch.scored("m-1", "Lathe A", 0.4),
            ]
        )

        found = await machine_reads.hybrid_search("a press", "press")

        assert [(hit.content.id_, hit.score, hit.rank) for hit in found] == [
            ("m-2", 0.9, 0),
            ("m-1", 0.4, 1),
        ]

    @staticmethod
    async def test__the_query__embeds_the_prefixed_meaning_and_uses_the_config(
        weaviate_double: WeaviateDouble,
        machine_reads: WeaviateReadRepository[Machine],
        embedder: Embedder,
    ) -> None:
        """The document type's prefix goes before the text embedded."""
        expected = await embedder.embed_query(["machine: a press"])

        await machine_reads.hybrid_search(
            "a press", "press", filters=[EqualityFilter(field="kind", value="press")]
        )

        hybrid = weaviate_double.collection.query.hybrid.await_args
        assert hybrid is not None
        assert hybrid.kwargs["vector"] == list(expected.embeddings[0])
        assert hybrid.kwargs["query"] == "press"
        assert (hybrid.kwargs["limit"], hybrid.kwargs["alpha"]) == (3, 0.7)
        assert hybrid.kwargs["filters"] == wcq.Filter.all_of([
            wcq.Filter.by_property("kind").equal("press")
        ])

    @staticmethod
    async def test__a_weight_of_zero__is_kept_not_replaced_by_the_config(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """A zero weight asks for keywords alone; it is not a missing weight."""
        await machine_reads.hybrid_search("a press", "press", limit=1, hybrid_weight=0.0)

        hybrid = weaviate_double.collection.query.hybrid.await_args
        assert hybrid is not None
        assert (hybrid.kwargs["limit"], hybrid.kwargs["alpha"]) == (1, 0.0)

    @staticmethod
    async def test__a_hit_without_a_score__is_refused(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """A rank without a score would mislead whoever reads it."""
        weaviate_double.collection.query.hybrid.return_value = QueryReturn(
            objects=[TestHybridSearch.scored("m-1", "Lathe A", None)]
        )

        with pytest.raises(ValueError, match="no score for m-1"):
            await machine_reads.hybrid_search("a lathe", "lathe")

    @staticmethod
    async def test__a_missing_collection__finds_nothing(
        weaviate_double: WeaviateDouble, machine_reads: WeaviateReadRepository[Machine]
    ) -> None:
        """No collection, no search sent."""
        weaviate_double.client.collections.exists.return_value = False

        found = await machine_reads.hybrid_search("a lathe", "lathe")

        assert found == []
        weaviate_double.collection.query.hybrid.assert_not_awaited()
