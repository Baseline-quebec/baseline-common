"""Dummy test classes for the multidigraph."""

from baseline_common.framework.multidigraph.graph_features import Edge, Node


class User(Node):
    """Represents a user node."""

    id_: str
    """The user's id."""
    age: int
    """The user's age."""
    loyalty_score: int
    """The user's loyalty score."""


class Flight(Node):
    """Represents a flight node."""

    id_: str
    """The flight's id."""
    duration: int
    """The flight's duration, in hours."""
    price: float
    """The flight's price."""
    other_node_attr: int
    """A number no other node type has."""
    carrier_id: str
    """The id of the carrier that operates it."""


class AircraftCarrier(Node):
    """Represents an aircraft carrier node."""

    id_: str
    """The carrier's id."""
    carrier_name: int
    """The carrier's name, a number here."""


class Booked(Edge[User | Flight]):
    """Represents a booked edge."""


class OperatedBy(Edge[AircraftCarrier | Flight]):
    """Represents an operated by edge."""


class Track(Edge[Node]):
    """A graded, dated track between two places."""

    grade: str
    """The track's grade."""
    year: int | None
    """The year the track was graded; None when unknown."""


class WeightedEdge(Edge[Node]):
    """An edge carrying a weight."""

    weight: float
    """The edge's weight."""
