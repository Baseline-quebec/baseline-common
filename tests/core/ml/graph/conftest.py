"""Fixtures for graph learning: the store schema, its features and its records."""

import pytest

from baseline_core.core.ml.graph.graph_feature_config import Feature, GraphFeatureConfig
from baseline_core.core.multidigraph.multidigraph_config import (
    EdgeConfig,
    MultiDiGraphConfig,
    NodeConfig,
)
from tests.core.multidigraph.store import Customer, Product, Purchased, Supplier, Supplies


@pytest.fixture
def store_config() -> MultiDiGraphConfig:
    """Customers purchase products; suppliers supply them."""
    customer = NodeConfig(id_="customer", node_class=Customer)
    product = NodeConfig(id_="product", node_class=Product)
    supplier = NodeConfig(id_="supplier", node_class=Supplier)

    return MultiDiGraphConfig(
        nodes=[customer, product, supplier],
        edges=[
            EdgeConfig(src_node=customer, relation="purchased", dst_node=product),
            EdgeConfig(src_node=supplier, relation="supplies", dst_node=product),
        ],
    )


@pytest.fixture
def store_features() -> GraphFeatureConfig:
    """Ages tripled by a transform that widens a feature; embeddings after the price."""
    return GraphFeatureConfig(
        features={
            "customer": [
                Feature(name="age", transform=lambda age: [age, age, age]),
                Feature(name="loyalty"),
            ],
            "product": [Feature(name="price"), Feature(name="embedding")],
            "supplier": [Feature(name="rating")],
        }
    )


@pytest.fixture
def store_nodes() -> list[Customer | Product | Supplier]:
    """Two customers, three products, one supplier."""
    return [
        Customer(id_="c1", age=25, loyalty=1.0),
        Customer(id_="c2", age=40, loyalty=10.0),
        Product(id_="p1", price=100.0, weight=1.0, embedding=[0.1, 0.2]),
        Product(id_="p2", price=200.0, weight=2.0, embedding=[0.3, 0.4]),
        Product(id_="p3", price=300.0, weight=3.0, embedding=[0.5, 0.6]),
        Supplier(id_="s1", rating=4),
    ]


@pytest.fixture
def store_edges(store_nodes: list[Customer | Product | Supplier]) -> list[Purchased | Supplies]:
    """Five purchases and three supplies."""
    c1, c2, p1, p2, p3, s1 = store_nodes
    purchases = [[c1, p1], [c1, p2], [c1, p3], [c2, p2], [c2, p3]]

    return [
        *[Purchased(src_node=src, dst_node=dst, quantity=1) for src, dst in purchases],
        *[Supplies(src_node=s1, dst_node=product) for product in [p1, p2, p3]],
    ]
