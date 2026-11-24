"""Tests for the graph schema."""

import pytest
from pydantic import ValidationError

from baseline_common.framework.multidigraph.graph_features import Node
from baseline_common.framework.multidigraph.multidigraph_config import (
    EdgeConfig,
    MultiDiGraphConfig,
    NodeConfig,
)
from tests.framework.multidigraph.store import Customer, Product, Supplier


class TestMultiDiGraphConfig:
    def test__when_node_types_are_looked_up__returns_the_one_of_the_node_class(
        self, store_config: MultiDiGraphConfig, product_config: NodeConfig
    ) -> None:
        """The product node type is the one whose node class is ``Product``."""
        assert store_config.get_node_config(Product) == product_config

    def test__when_a_node_class_is_undescribed__raises_value_error(
        self, store_config: MultiDiGraphConfig
    ) -> None:
        """A class no node type describes has no node config."""
        with pytest.raises(ValueError, match="No node config"):
            store_config.get_node_config(Node)

    def test__when_relations_are_looked_up__returns_the_one_of_the_end_classes(
        self, store_config: MultiDiGraphConfig
    ) -> None:
        """Customer to product is the ``purchased`` relation."""
        assert store_config.get_edge_config(Customer, Product).relation == "purchased"

    def test__when_a_relation_is_looked_up_backwards__raises_value_error(
        self, store_config: MultiDiGraphConfig
    ) -> None:
        """Product to customer is not ``purchased`` read backwards."""
        with pytest.raises(ValueError, match="No edge config from Product to Customer"):
            store_config.get_edge_config(Product, Customer)

    def test__when_two_node_types_are_of_one_class__raises_validation_error(
        self, customer_config: NodeConfig
    ) -> None:
        """A node class described twice would make its node type ambiguous."""
        twin = NodeConfig(id_="buyer", node_class=Customer)

        with pytest.raises(ValidationError, match="Node classes must be unique"):
            MultiDiGraphConfig(nodes=[customer_config, twin])

    def test__when_two_relations_join_two_node_types__raises_validation_error(
        self, customer_config: NodeConfig, product_config: NodeConfig
    ) -> None:
        """A second relation from customers to products would make the lookup ambiguous."""
        edges = [
            EdgeConfig(src_node=customer_config, relation=relation, dst_node=product_config)
            for relation in ["purchased", "viewed"]
        ]

        with pytest.raises(ValidationError, match="Edge relations must be unique"):
            MultiDiGraphConfig(nodes=[customer_config, product_config], edges=edges)

    def test__when_the_schema_is_projected_to_networkx__returns_a_graph_of_node_types(
        self, store_config: MultiDiGraphConfig
    ) -> None:
        """The schema projects to networkx like any graph: one node per node type."""
        graph = store_config.networkx_graph

        assert sorted(graph.nodes) == ["customer", "product", "supplier"]
        assert sorted(graph.edges(keys=False)) == [
            ("customer", "product"),
            ("supplier", "product"),
        ]
        assert graph.nodes["supplier"]["node_class"] is Supplier
