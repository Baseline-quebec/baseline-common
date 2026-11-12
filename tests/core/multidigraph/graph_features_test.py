"""Tests for the multidigraph features module."""

import pytest
from pydantic import ValidationError

from baseline_core.core.multidigraph.graph_features import Edge, Node, SubgraphQuery
from tests.core.multidigraph.flights import AircraftCarrier, Booked, Flight, OperatedBy, User


class TestNode:
    """Test cases for the Node class."""

    def test__node__returns_node_with_extra_fields(self, user1_fixture: User) -> None:
        """Test that Node returns a node with extra fields."""
        assert isinstance(user1_fixture, Node)
        assert isinstance(user1_fixture, User)

        expected_age = 25
        expected_loyalty_score = 1
        assert user1_fixture.id_ == "u1"
        assert user1_fixture.age == expected_age
        assert user1_fixture.loyalty_score == expected_loyalty_score

    def test__node__when_subclass_instantiated__feature_type_str_returns_subclass_name(
        self, user1_fixture: User
    ) -> None:
        """Test that Node returns the correct feature_type_str based on the subclass."""
        node: User = user1_fixture

        result: str = node.feature_type_str

        assert result == "User"

    def test__node__config_is_frozen(self) -> None:
        """Tripwire: the whole node-index invariant rests on nodes being frozen."""
        assert Node.model_config.get("frozen") is True

    def test__node__when_attribute_reassigned__raises(self, user1_fixture: User) -> None:
        """A frozen node rejects in-place mutation, so indexed values cannot drift."""
        with pytest.raises(ValidationError):
            user1_fixture.id_ = "mutated"  # type: ignore[misc]  # the write a frozen node refuses


class TestEdge:
    """Test cases for the Edge class."""

    @pytest.fixture
    def edge_fixture(self, user1_fixture: User, flight1_fixture: Flight) -> Booked:
        """Edge fixture."""
        return Booked(src_node=user1_fixture, dst_node=flight1_fixture)

    @pytest.fixture
    def operated_by_edge_fixture(
        self, aircraft_carrier_fixture: AircraftCarrier, flight1_fixture: Flight
    ) -> OperatedBy:
        """OperatedBy edge fixture."""
        return OperatedBy(src_node=aircraft_carrier_fixture, dst_node=flight1_fixture)

    def test__edge__returns_edge(
        self, edge_fixture: Booked, user1_fixture: User, flight1_fixture: Flight
    ) -> None:
        """Test the ability to create an edge."""
        assert isinstance(edge_fixture, Edge)
        assert isinstance(edge_fixture, Booked)
        assert edge_fixture.feature_type_str == "Booked"
        assert edge_fixture.src_node == user1_fixture
        assert edge_fixture.dst_node == flight1_fixture

    def test__edge__when_subclass_instantiated__feature_type_str_returns_subclass_name(
        self, edge_fixture: Booked
    ) -> None:
        """Test that Edge returns the correct feature_type_str based on the subclass."""
        edge: Booked = edge_fixture

        result: str = edge.feature_type_str

        assert result == "Booked"

    def test__edge__when_different_subclass_with_same_endpoints__not_equal(
        self, edge_fixture: Booked, operated_by_edge_fixture: OperatedBy
    ) -> None:
        """Test that two edges with the same endpoints but different subclasses are not equal."""
        booked_edge: Booked = edge_fixture
        operated_by_edge: OperatedBy = operated_by_edge_fixture

        assert booked_edge != operated_by_edge  # type: ignore[comparison-overlap]  # the point

    def test__edge__config_is_frozen(self) -> None:
        """Tripwire: the edge-incidence-index invariant rests on edges being frozen."""
        assert Edge.model_config.get("frozen") is True

    def test__edge__when_attribute_reassigned__raises(self, edge_fixture: Booked) -> None:
        """A frozen edge rejects in-place mutation, so indexed endpoints cannot drift."""
        with pytest.raises(ValidationError):
            edge_fixture.src_node = edge_fixture.dst_node  # type: ignore[misc]  # refused: frozen


class TestSubgraphQuery:
    """Test cases for the query shape."""

    def test__query__with_no_criterion__is_refused(self) -> None:
        """A query that would match everything by accident is a bug, not a search."""
        with pytest.raises(ValidationError, match="at least one matching criterion"):
            SubgraphQuery()

    def test__query__with_a_crossed_window__is_refused(self) -> None:
        """An empty window would silently drop every edge."""
        with pytest.raises(ValidationError, match="Empty window"):
            SubgraphQuery(node_ids=["a"], edge_bounds={"year": (2022, 2020)})
