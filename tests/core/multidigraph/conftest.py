"""Fixtures for the multidigraph test suite."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pytest

from baseline_common.core.multidigraph.graph_features import Edge, Node
from baseline_common.core.multidigraph.multidigraph import MultiDiGraph
from baseline_common.core.multidigraph.multidigraph_config import (
    EdgeConfig,
    MultiDiGraphConfig,
    NodeConfig,
)
from tests.core.multidigraph.flights import (
    AircraftCarrier,
    Booked,
    Flight,
    OperatedBy,
    Track,
    User,
)
from tests.core.multidigraph.store import Customer, Product, Purchased, Supplier, Supplies


if TYPE_CHECKING:
    import networkx as nx


@pytest.fixture
def user1_fixture() -> User:
    """User 1 fixture."""
    return User(id_="u1", age=25, loyalty_score=1)


@pytest.fixture
def user2_fixture() -> User:
    """User 2 fixture."""
    return User(id_="u2", age=40, loyalty_score=10)


@pytest.fixture
def flight1_fixture() -> Flight:
    """Flight 1 fixture with vector embedding."""
    return Flight(id_="f1", duration=1, price=100, other_node_attr=1, carrier_id="ac1")


@pytest.fixture
def flight2_fixture() -> Flight:
    """Flight 2 fixture with vector embedding."""
    return Flight(id_="f2", duration=2, price=200, other_node_attr=2, carrier_id="ac2")


@pytest.fixture
def flight3_fixture() -> Flight:
    """Flight 3 fixture with vector embedding."""
    return Flight(id_="f3", duration=3, price=300, other_node_attr=3, carrier_id="ac3")


@pytest.fixture
def aircraft_carrier_fixture() -> AircraftCarrier:
    """Aircraft carrier fixture."""
    return AircraftCarrier(id_="ac1", carrier_name=123)


@pytest.fixture
def empty_graph_fixture() -> MultiDiGraph[Node, Edge[Any]]:
    """Empty graph fixture."""
    return MultiDiGraph(nodes=[], edges=[])


@pytest.fixture
def node_data_fixture(
    user1_fixture: User,
    user2_fixture: User,
    flight1_fixture: Flight,
    flight2_fixture: Flight,
    flight3_fixture: Flight,
    aircraft_carrier_fixture: AircraftCarrier,
) -> dict[type[Node], list[Node]]:
    """Node data fixture."""
    return {
        User: [user1_fixture, user2_fixture],
        Flight: [flight1_fixture, flight2_fixture, flight3_fixture],
        AircraftCarrier: [aircraft_carrier_fixture],
    }


@pytest.fixture
def booked_edge_data_fixture(
    user1_fixture: User,
    user2_fixture: User,
    flight1_fixture: Flight,
    flight2_fixture: Flight,
    flight3_fixture: Flight,
) -> list[Booked]:
    """Booked edge data fixture."""
    return [
        Booked(src_node=user1_fixture, dst_node=flight1_fixture),
        Booked(src_node=user1_fixture, dst_node=flight2_fixture),
        Booked(src_node=user1_fixture, dst_node=flight3_fixture),
        Booked(src_node=user2_fixture, dst_node=flight2_fixture),
        Booked(src_node=user2_fixture, dst_node=flight3_fixture),
        Booked(src_node=flight1_fixture, dst_node=user1_fixture),
        Booked(src_node=flight2_fixture, dst_node=user1_fixture),
        Booked(src_node=flight3_fixture, dst_node=user1_fixture),
        Booked(src_node=flight2_fixture, dst_node=user2_fixture),
        Booked(src_node=flight3_fixture, dst_node=user2_fixture),
    ]


@pytest.fixture
def operated_by_edge_data_fixture(
    aircraft_carrier_fixture: AircraftCarrier,
    flight1_fixture: Flight,
    flight2_fixture: Flight,
    flight3_fixture: Flight,
) -> list[OperatedBy]:
    """Operated by edge data fixture."""
    return [
        OperatedBy(src_node=aircraft_carrier_fixture, dst_node=flight1_fixture),
        OperatedBy(src_node=aircraft_carrier_fixture, dst_node=flight2_fixture),
        OperatedBy(src_node=aircraft_carrier_fixture, dst_node=flight3_fixture),
    ]


@pytest.fixture
def edge_data_fixture(
    booked_edge_data_fixture: list[Booked], operated_by_edge_data_fixture: list[OperatedBy]
) -> dict[type[Edge[Any]], Sequence[Edge[Any]]]:
    """Edge data fixture."""
    return {Booked: booked_edge_data_fixture, OperatedBy: operated_by_edge_data_fixture}


@pytest.fixture
def multidigraph_fixture(
    node_data_fixture: dict[type[Node], list[Node]],
    edge_data_fixture: dict[type[Edge[Any]], Sequence[Edge[Any]]],
) -> MultiDiGraph[Node, Edge[Any]]:
    """User-flight graph fixture with booked edges."""
    return MultiDiGraph(
        nodes=[
            *node_data_fixture[User],
            *node_data_fixture[Flight],
            *node_data_fixture[AircraftCarrier],
        ],
        edges=[*edge_data_fixture[Booked], *edge_data_fixture[OperatedBy]],
    )


@pytest.fixture
def unstrongly_connected_multidigraph_fixture(
    multidigraph_fixture: MultiDiGraph[Node, Edge[Any]],
) -> MultiDiGraph[Node, Edge[Any]]:
    """Unstrongly connected multidigraph fixture."""
    u = User(id_="u3", age=3, loyalty_score=3)
    f = Flight(id_="f4", duration=4, price=400, other_node_attr=4, carrier_id="ac4")
    multidigraph_fixture.add_bidirectional_edge(Booked(src_node=u, dst_node=f))

    return multidigraph_fixture


@pytest.fixture
def networkx_multidigraph_fixture(
    multidigraph_fixture: MultiDiGraph[Node, Edge[Any]],
) -> "nx.MultiDiGraph[str]":
    """NetworkX multidigraph fixture."""
    return multidigraph_fixture.networkx_graph


@pytest.fixture
def line_graph_fixture() -> MultiDiGraph[Node, Track]:
    """A line a—b—c—d of graded, dated tracks, plus an island no track reaches."""
    north_a = Node(id_="a", region="north")
    north_b = Node(id_="b", region="north")
    south_c = Node(id_="c", region="south")
    south_d = Node(id_="d", region="south")
    island = Node(id_="island", region="north")
    return MultiDiGraph(
        nodes=[north_a, north_b, south_c, south_d, island],
        edges=[
            Track(src_node=north_a, dst_node=north_b, grade="main", year=2020),
            Track(src_node=north_b, dst_node=south_c, grade="branch", year=2021),
            Track(src_node=south_c, dst_node=south_d, grade="branch", year=None),
        ],
    )


@pytest.fixture
def customer_config() -> NodeConfig:
    """The customer node type."""
    return NodeConfig(id_="customer", node_class=Customer)


@pytest.fixture
def product_config() -> NodeConfig:
    """The product node type."""
    return NodeConfig(id_="product", node_class=Product)


@pytest.fixture
def supplier_config() -> NodeConfig:
    """The supplier node type."""
    return NodeConfig(id_="supplier", node_class=Supplier)


@pytest.fixture
def store_config(
    customer_config: NodeConfig, product_config: NodeConfig, supplier_config: NodeConfig
) -> MultiDiGraphConfig:
    """Customers purchase products; suppliers supply them."""
    return MultiDiGraphConfig(
        nodes=[customer_config, product_config, supplier_config],
        edges=[
            EdgeConfig(src_node=customer_config, relation="purchased", dst_node=product_config),
            EdgeConfig(src_node=supplier_config, relation="supplies", dst_node=product_config),
        ],
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
