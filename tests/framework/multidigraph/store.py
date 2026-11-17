"""A small store: customers purchase products, suppliers supply them."""

from baseline_common.framework.multidigraph.graph_features import Edge, Node


class Customer(Node):
    """A customer."""

    age: int
    """The customer's age."""
    loyalty: float
    """The customer's loyalty."""


class Product(Node):
    """A product, with an embedding of its description."""

    price: float
    """The product's price."""
    weight: float
    """The product's weight."""
    embedding: list[float]
    """The embedding of the product's description."""


class Supplier(Node):
    """A supplier."""

    rating: int
    """The supplier's rating."""


class Purchased(Edge[Customer | Product]):
    """A customer purchased a product."""

    quantity: int
    """How many units the customer bought."""


class Supplies(Edge[Supplier | Product]):
    """A supplier supplies a product."""
