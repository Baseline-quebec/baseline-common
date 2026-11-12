"""Tests for the HeteroData converter."""

import pytest
import torch

from baseline_core.core.ml.graph.graph_feature_config import Feature, GraphFeatureConfig
from baseline_core.core.ml.graph.hetero_data_converter import HeteroDataConverter
from baseline_core.core.multidigraph.graph_builder import GraphBuilder
from baseline_core.core.multidigraph.graph_features import Edge, Node
from baseline_core.core.multidigraph.multidigraph import MultiDiGraph
from baseline_core.core.multidigraph.multidigraph_config import MultiDiGraphConfig
from tests.core.multidigraph.store import Customer, Product, Purchased, Supplier, Supplies


class TestHeteroDataConverter:
    @pytest.fixture
    def converter(
        self, store_config: MultiDiGraphConfig, store_features: GraphFeatureConfig
    ) -> HeteroDataConverter:
        """A converter for the store schema and its features."""
        return HeteroDataConverter(store_config, store_features)

    @pytest.fixture
    def store_graph(
        self,
        store_config: MultiDiGraphConfig,
        store_nodes: list[Customer | Product | Supplier],
        store_edges: list[Purchased | Supplies],
    ) -> MultiDiGraph[Customer | Product | Supplier, Purchased | Supplies]:
        """The store, built against its schema."""
        return GraphBuilder(store_config).build(store_nodes, store_edges)

    def test__node_types__are_keyed_by_the_schema_id(
        self,
        converter: HeteroDataConverter,
        store_graph: MultiDiGraph[Customer | Product | Supplier, Purchased | Supplies],
    ) -> None:
        """Each node type of the schema becomes a node type of the data."""
        data = converter.to_hetero_data(store_graph)

        assert sorted(data.node_types) == ["customer", "product", "supplier"]
        assert sorted(data.edge_types) == [
            ("customer", "purchased", "product"),
            ("supplier", "supplies", "product"),
        ]

    def test__features__are_transformed_and_flattened(
        self,
        converter: HeteroDataConverter,
        store_graph: MultiDiGraph[Customer | Product | Supplier, Purchased | Supplies],
    ) -> None:
        """Ages are tripled by their transform; embeddings are flattened after the price."""
        data = converter.to_hetero_data(store_graph)

        assert torch.equal(
            data["customer"].x, torch.tensor([[25.0, 25.0, 25.0, 1.0], [40.0, 40.0, 40.0, 10.0]])
        )
        assert torch.allclose(
            data["product"].x,
            torch.tensor([[100.0, 0.1, 0.2], [200.0, 0.3, 0.4], [300.0, 0.5, 0.6]]),
        )
        assert data["supplier"].num_nodes == 1

    def test__edge_indexes__count_rows_within_each_node_type(
        self,
        converter: HeteroDataConverter,
        store_graph: MultiDiGraph[Customer | Product | Supplier, Purchased | Supplies],
    ) -> None:
        """Purchases point from customer rows to product rows, in graph order."""
        data = converter.to_hetero_data(store_graph)

        purchased = data["customer", "purchased", "product"].edge_index
        supplies = data["supplier", "supplies", "product"].edge_index
        assert purchased.tolist() == [[0, 0, 0, 1, 1], [0, 1, 2, 1, 2]]
        assert supplies.tolist() == [[0, 0, 0], [0, 1, 2]]

    def test__an_empty_graph__gives_empty_data(self, converter: HeteroDataConverter) -> None:
        """No nodes, no node types; no edges, no edge types."""
        data = converter.to_hetero_data(MultiDiGraph())

        assert data.node_types == []
        assert data.edge_types == []

    def test__an_undescribed_node__is_refused(self, converter: HeteroDataConverter) -> None:
        """A node the schema does not describe has no feature list to read."""
        graph: MultiDiGraph[Node, Edge[Node]] = MultiDiGraph(nodes=[Node(id_="x")])

        with pytest.raises(ValueError, match="No node config"):
            converter.to_hetero_data(graph)

    def test__an_edge_against_a_relation__is_refused(
        self, converter: HeteroDataConverter, store_nodes: list[Customer | Product | Supplier]
    ) -> None:
        """A purchase from a product to a customer goes against the relation's direction."""
        backwards = Purchased(src_node=store_nodes[2], dst_node=store_nodes[0], quantity=1)
        graph: MultiDiGraph[Customer | Product | Supplier, Purchased] = MultiDiGraph(
            nodes=store_nodes, edges=[backwards]
        )

        with pytest.raises(ValueError, match="No edge config from Product to Customer"):
            converter.to_hetero_data(graph)

    def test__features_for_another_set_of_node_types__are_refused(
        self, store_config: MultiDiGraphConfig
    ) -> None:
        """Every node type of the schema, and only those, has its features."""
        features = GraphFeatureConfig(features={"customer": [Feature(name="age")]})

        with pytest.raises(ValueError, match="exactly the node types"):
            HeteroDataConverter(store_config, features)

    def test__feature_vector__reads_the_features_in_order(
        self, store_nodes: list[Customer | Product | Supplier], store_features: GraphFeatureConfig
    ) -> None:
        """A product reads as its price, then its embedding."""
        vector = HeteroDataConverter.feature_vector(
            store_nodes[2], store_features.features["product"]
        )

        assert vector == pytest.approx([100.0, 0.1, 0.2])
