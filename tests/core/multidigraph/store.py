"""A small store: customers purchase products, suppliers supply them."""

from baseline_common.core.multidigraph.graph_features import Edge, Node


class Customer(Node):
    """A customer."""

    age: int
    loyalty: float


class Product(Node):
    """A product, with an embedding of its description."""

    price: float
    weight: float
    embedding: list[float]


class Supplier(Node):
    """A supplier."""

    rating: int


class Purchased(Edge[Customer | Product]):
    """A customer purchased a product."""

    quantity: int


class Supplies(Edge[Supplier | Product]):
    """A supplier supplies a product."""
