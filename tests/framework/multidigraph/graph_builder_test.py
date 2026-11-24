"""Tests for the graph builder."""

from typing import TYPE_CHECKING

import pytest

from baseline_common.framework.multidigraph.graph_builder import GraphBuilder
from baseline_common.framework.multidigraph.graph_features import Edge, Node
from baseline_common.framework.multidigraph.multidigraph_config import MultiDiGraphConfig
from tests.framework.multidigraph.store import Customer, Product, Purchased, Supplier, Supplies


if TYPE_CHECKING:
    from baseline_common.framework.multidigraph.multidigraph import MultiDiGraph


class TestGraphBuilder:
    @pytest.fixture
    def builder(self, store_config: MultiDiGraphConfig) -> GraphBuilder:
        """A builder for the store schema."""
        return GraphBuilder(store_config)

    def test__when_records_are_built__returns_every_record(
        self,
        builder: GraphBuilder,
        store_nodes: list[Customer | Product | Supplier],
        store_edges: list[Purchased | Supplies],
    ) -> None:
        """Six nodes and eight edges go in, six nodes and eight edges come out, types kept."""
        graph = builder.build(store_nodes, store_edges)

        assert graph.nodes == store_nodes
        assert len(graph.edges) == len(store_edges)
        assert [type(edge) for edge in graph.edges] == [type(edge) for edge in store_edges]

    def test__when_records_are_built__returns_each_edge_with_its_relation(
        self,
        builder: GraphBuilder,
        store_nodes: list[Customer | Product | Supplier],
        store_edges: list[Purchased | Supplies],
    ) -> None:
        """Purchases carry ``purchased``, supplies carry ``supplies``, other fields kept."""
        graph = builder.build(store_nodes, store_edges)

        relations = [edge.relation for edge in graph.edges]  # type: ignore[union-attr]  # an extra
        first = graph.edges[0]
        assert relations == ["purchased"] * 5 + ["supplies"] * 3
        assert isinstance(first, Purchased)
        assert first.quantity == 1

    def test__when_an_edge_holds_a_copy_of_a_node__returns_it_on_the_graph_own_node(
        self, builder: GraphBuilder, store_nodes: list[Customer | Product | Supplier]
    ) -> None:
        """An edge record holding a copy of a node is joined to the graph's instance."""
        customer, product = store_nodes[0], store_nodes[2]
        copied = Purchased(src_node=customer.model_copy(), dst_node=product, quantity=2)

        graph = builder.build(store_nodes, [copied])

        assert graph.edges[0].src_node is customer
        assert graph.edges_of("c1") == graph.edges

    def test__when_a_builder_builds_twice__returns_two_separate_graphs(
        self,
        builder: GraphBuilder,
        store_nodes: list[Customer | Product | Supplier],
        store_edges: list[Purchased | Supplies],
    ) -> None:
        """A build keeps nothing from the previous one."""
        customers = store_nodes[:2]
        no_edges: list[Purchased] = []
        first = builder.build(store_nodes, store_edges)

        second = builder.build(customers, no_edges)

        assert first.nodes == store_nodes
        assert second.nodes == customers
        assert second.edges == []

    def test__when_one_product_is_purchased_twice__returns_two_edges(
        self, builder: GraphBuilder
    ) -> None:
        """A customer buying a product twice gives two ``purchased`` edges on the same pair."""
        customer = Customer(id_="c1", age=25, loyalty=1.0)
        product = Product(id_="p1", price=100.0, weight=1.0, embedding=[0.1, 0.2])
        purchases = [
            Purchased(src_node=customer, dst_node=product, quantity=quantity)
            for quantity in [1, 2]
        ]

        graph = builder.build([customer, product], purchases)

        assert [edge.quantity for edge in graph.edges] == [1, 2]

    def test__when_one_purchase_is_given_twice__raises_value_error(
        self, builder: GraphBuilder
    ) -> None:
        """The same record twice is one edge too many."""
        customer = Customer(id_="c1", age=25, loyalty=1.0)
        product = Product(id_="p1", price=100.0, weight=1.0, embedding=[0.1, 0.2])
        purchase = Purchased(src_node=customer, dst_node=product, quantity=1)

        with pytest.raises(ValueError, match="The edge c1 -> p1 is given twice"):
            builder.build([customer, product], [purchase, purchase])

    def test__when_nothing_is_built__returns_an_empty_graph(self, builder: GraphBuilder) -> None:
        """No records, no nodes, no edges."""
        graph: MultiDiGraph[Node, Edge[Node]] = builder.build([], [])

        assert graph.nodes == []
        assert graph.edges == []

    def test__when_a_node_is_of_an_undescribed_class__raises_value_error(
        self, builder: GraphBuilder
    ) -> None:
        """A node the schema does not describe has no place in the graph."""
        with pytest.raises(ValueError, match="No node config for the node class Node"):
            builder.build([Node(id_="x")], [])

    def test__when_an_edge_goes_against_a_relation__raises_value_error(
        self, builder: GraphBuilder, store_nodes: list[Customer | Product | Supplier]
    ) -> None:
        """A purchase from a product to a customer goes against the relation's direction."""
        backwards = Purchased(src_node=store_nodes[2], dst_node=store_nodes[0], quantity=1)

        with pytest.raises(ValueError, match="No edge config from Product to Customer"):
            builder.build(store_nodes, [backwards])

    def test__when_an_edge_goes_to_a_missing_node__raises_value_error(
        self, builder: GraphBuilder, store_nodes: list[Customer | Product | Supplier]
    ) -> None:
        """An edge whose end is not among the nodes is refused."""
        purchase = Purchased(src_node=store_nodes[0], dst_node=store_nodes[2], quantity=1)

        with pytest.raises(ValueError, match="Node with id p1 not found"):
            builder.build(store_nodes[:2], [purchase])
