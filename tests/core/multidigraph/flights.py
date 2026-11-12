"""Dummy test classes for the multidigraph."""

from baseline_common.core.multidigraph.graph_features import Edge, Node


class User(Node):
    """Represents a user node."""

    id_: str
    age: int
    loyalty_score: int


class Flight(Node):
    """Represents a flight node."""

    id_: str
    duration: int
    price: float
    other_node_attr: int
    carrier_id: str


class AircraftCarrier(Node):
    """Represents an aircraft carrier node."""

    id_: str
    carrier_name: int


class Booked(Edge[User | Flight]):
    """Represents a booked edge."""


class OperatedBy(Edge[AircraftCarrier | Flight]):
    """Represents an operated by edge."""


class Track(Edge[Node]):
    """A graded, dated track between two places."""

    grade: str
    year: int | None


class WeightedEdge(Edge[Node]):
    """An edge carrying a weight."""

    weight: float
