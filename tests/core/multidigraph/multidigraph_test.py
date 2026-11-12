"""Tests for the multidigraph module."""

from typing import Any

import networkx as nx
import pytest
from pydantic import ValidationError

from baseline_common.core.multidigraph.graph_features import Edge, Expansion, Node, SubgraphQuery
from baseline_common.core.multidigraph.multidigraph import MultiDiGraph
from tests.core.multidigraph.flights import Track, WeightedEdge


class TestMultiDiGraph:
    """Test cases for the MultiDiGraph class."""

    def test__multidigraph__returns_multidigraph(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to create a MultiDiGraph."""
        assert isinstance(multidigraph_fixture, MultiDiGraph)

    def test__add_node__adds_node_to_graph(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to add a node to the graph."""
        a_node = Node(id_="1")

        assert len(empty_graph_fixture.nodes) == 0
        node_id = empty_graph_fixture.add_node(a_node)
        assert len(empty_graph_fixture.nodes) == 1
        assert empty_graph_fixture.get_node_by_id(node_id) == a_node

    def test__add_edge__adds_edge_to_graph(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to add an edge to the graph."""
        a_edge: Edge[Node] = Edge(src_node=Node(id_="1"), dst_node=Node(id_="2"))

        empty_graph_fixture.add_edge(a_edge)

        expected_num_edges = 1
        assert len(empty_graph_fixture.edges) == expected_num_edges
        assert empty_graph_fixture.edges[0] == a_edge

    def test__add_bidirectional_edge__adds_edges_to_graph(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to add a bidirectional edge to the graph."""
        a_edge: Edge[Node] = Edge(src_node=Node(id_="1"), dst_node=Node(id_="2"))

        empty_graph_fixture.add_bidirectional_edge(a_edge)

        expected_num_edges = 2
        assert len(empty_graph_fixture.edges) == expected_num_edges

    def test__get_node_by_id__returns_node(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to get a node by its id."""
        a_node = Node(id_="1")
        node_id = empty_graph_fixture.add_node(a_node)

        result = empty_graph_fixture.get_node_by_id(node_id)

        assert result == a_node

    def test__get_node_by_id__raises_error_when_not_found(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test that getting a non-existent node raises ValueError."""
        with pytest.raises(ValueError, match="Node with id nonexistent not found"):
            empty_graph_fixture.get_node_by_id("nonexistent")

    def test__to_networkx_graph__returns_multidigraph(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to create a NetworkX graph from a MultiDiGraph."""
        result_graph = multidigraph_fixture.networkx_graph

        assert isinstance(result_graph, nx.MultiDiGraph)

        assert {node.id_ for node in multidigraph_fixture.nodes} == set(result_graph.nodes)

        assert {(edge.src_node.id_, edge.dst_node.id_) for edge in multidigraph_fixture.edges} == {
            (e[0], e[1]) for e in result_graph.edges()
        }

    def test__get_strongly_connected_components__returns_components(
        self, unstrongly_connected_multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to get the strongly connected components of a multidigraph."""
        components = unstrongly_connected_multidigraph_fixture.get_strongly_connected_components()

        n_components = 3
        assert len(components) == n_components
        assert set(components[0]) == {"u1", "u2", "f1", "f2", "f3"}
        assert set(components[1]) == {"u3", "f4"}
        assert set(components[2]) == {"ac1"}
        assert unstrongly_connected_multidigraph_fixture.get_is_strongly_connected() is False

    def test__get_is_strongly_connected__returns_true_when_connected(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test that is_strongly_connected returns True for a connected graph."""
        node_1 = Node(id_="1")
        node_2 = Node(id_="2")
        node_3 = Node(id_="3")
        empty_graph_fixture.add_node(node_1)
        empty_graph_fixture.add_node(node_2)
        empty_graph_fixture.add_node(node_3)
        empty_graph_fixture.add_bidirectional_edge(Edge(src_node=node_1, dst_node=node_2))
        empty_graph_fixture.add_bidirectional_edge(Edge(src_node=node_2, dst_node=node_3))
        assert empty_graph_fixture.get_is_strongly_connected() is True

    def test__relabel_nodes__returns_new_node_ids(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to relabel the nodes in the MultiDiGraph."""
        init_node_ids = {node.id_ for node in multidigraph_fixture.nodes}

        relabeled_multidigraph = multidigraph_fixture.relabel_nodes()

        new_node_ids = {node.id_ for node in relabeled_multidigraph.nodes}
        assert new_node_ids != init_node_ids

    def test__from_networkx_graph__returns_multidigraph(
        self, networkx_multidigraph_fixture: "nx.MultiDiGraph[str]"
    ) -> None:
        """Test the ability to create a MultiDiGraph from a NetworkX graph."""
        result_multidigraph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph.from_networkx_graph(
            networkx_multidigraph_fixture
        )

        assert isinstance(result_multidigraph, MultiDiGraph)

    def test__get_strongly_connected_components_data__returns_data(self) -> None:
        """Test getting component data from connected components."""
        connected_components = [["1", "2", "3"], ["4", "5"]]

        components_set, components_node_ids, components_node_idx = (
            MultiDiGraph.get_strongly_connected_components_data(connected_components)
        )

        assert components_set[0] == {"1", "2", "3"}
        assert components_set[1] == {"4", "5"}
        assert components_node_ids[0] == ["1", "2", "3"]
        assert components_node_ids[1] == ["4", "5"]
        assert components_node_idx[0] == [0, 1, 2]
        assert components_node_idx[1] == [3, 4]

    def test__group_nodes_by__returns_nodes_grouped_by_attribute(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test grouping nodes by an attribute."""
        empty_graph_fixture.add_node(Node(id_="1", category="A"))
        empty_graph_fixture.add_node(Node(id_="2", category="B"))
        empty_graph_fixture.add_node(Node(id_="3", category="A"))
        empty_graph_fixture.add_node(Node(id_="4", category="B"))

        result = empty_graph_fixture.group_nodes_by("category")

        n_groups = 2
        n_nodes_per_group = 2
        assert len(result) == n_groups
        assert len(result["A"]) == n_nodes_per_group
        assert len(result["B"]) == n_nodes_per_group

    def test__group_nodes_by__returns_single_group_when_all_same(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test grouping nodes when all have the same attribute value."""
        empty_graph_fixture.add_node(Node(id_="1"))
        empty_graph_fixture.add_node(Node(id_="2"))
        empty_graph_fixture.add_node(Node(id_="3"))

        result = empty_graph_fixture.group_nodes_by("feature_type_str")

        n_nodes = 3
        assert len(result) == 1
        assert len(result["Node"]) == n_nodes

    def test__group_nodes_by__raises_error_when_attribute_absent_from_all_nodes(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test that group_nodes_by raises when no indexed node carries the attr."""
        empty_graph_fixture.add_node(Node(id_="1", category="A"))
        empty_graph_fixture.add_node(Node(id_="2"))  # Missing the 'category' attribute

        with pytest.raises(
            ValueError, match="Node type attribute unknown_attr is not found in the node"
        ):
            empty_graph_fixture.group_nodes_by("unknown_attr")

    def test__group_nodes_by__groups_only_nodes_carrying_the_attribute(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test that grouping by a partially-present attr includes only carriers."""
        node_with_attr = Node(id_="1", category="A")
        empty_graph_fixture.add_node(node_with_attr)
        empty_graph_fixture.add_node(Node(id_="2"))  # Missing the 'category' attribute

        result = empty_graph_fixture.group_nodes_by("category")

        assert result == {"A": [node_with_attr]}

    def test__group_nodes_by__groups_by_node_type(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test grouping nodes by the computed node_type field."""
        empty_graph_fixture.add_node(Node(id_="1"))
        empty_graph_fixture.add_node(Node(id_="2"))

        result = empty_graph_fixture.group_nodes_by("feature_type_str")

        n_nodes = 2
        assert len(result) == 1
        assert "Node" in result
        assert len(result["Node"]) == n_nodes

    def test__filter_nodes_by__returns_filtered_nodes(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test filtering nodes by an attribute value."""
        empty_graph_fixture.add_node(Node(id_="1", category="A"))
        empty_graph_fixture.add_node(Node(id_="2", category="B"))
        empty_graph_fixture.add_node(Node(id_="3", category="A"))

        result = empty_graph_fixture.filter_nodes_by("category", "A")

        n_filtered_nodes = 2
        assert len(result) == n_filtered_nodes
        assert all(node.model_dump().get("category") == "A" for node in result)

    def test__filter_nodes_by__returns_empty_when_no_match(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test filtering nodes returns empty list when no nodes match."""
        empty_graph_fixture.add_node(Node(id_="1", category="A"))
        empty_graph_fixture.add_node(Node(id_="2", category="B"))

        result = empty_graph_fixture.filter_nodes_by("category", "C")

        assert len(result) == 0

    def test__remove_node__removes_node_from_graph(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to remove a node from the graph."""
        a_node = Node(id_="1")
        empty_graph_fixture.add_node(a_node)
        assert len(empty_graph_fixture.nodes) == 1

        empty_graph_fixture.remove_node(a_node)

        assert len(empty_graph_fixture.nodes) == 0

    def test__remove_edge__removes_edge_from_graph(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test the ability to remove an edge from the graph."""
        node1 = Node(id_="1")
        node2 = Node(id_="2")
        a_edge: Edge[Node] = Edge(src_node=node1, dst_node=node2)
        empty_graph_fixture.add_edge(a_edge)
        assert len(empty_graph_fixture.edges) == 1

        empty_graph_fixture.remove_edge(a_edge)

        assert len(empty_graph_fixture.edges) == 0

    def test__validate_unique_nodes__raises_error_on_duplicate(self) -> None:
        """Test that creating a MultiDiGraph with duplicate nodes raises ValueError."""
        node1 = Node(id_="1")
        node2 = Node(id_="1")

        with pytest.raises(ValueError, match="Node types must be unique"):
            MultiDiGraph(nodes=[node1, node2])

    def test__validate_unique_edges__raises_error_on_duplicate(self) -> None:
        """Test that creating a MultiDiGraph with duplicate edges raises ValueError."""
        node1 = Node(id_="1")
        node2 = Node(id_="2")
        edge1: Edge[Node] = Edge(src_node=node1, dst_node=node2)
        edge2: Edge[Node] = Edge(src_node=node1, dst_node=node2)

        with pytest.raises(ValueError, match="Edge relations must be unique"):
            MultiDiGraph(nodes=[node1, node2], edges=[edge1, edge2])

    def test__merge_multidigraph__merges_two_graphs(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test merging two MultiDiGraphs together."""
        node1 = Node(id_="a")
        node2 = Node(id_="b")
        empty_graph_fixture.add_node(node1)
        empty_graph_fixture.add_node(node2)
        bidirectional_edge: Edge[Node] = Edge(src_node=node1, dst_node=node2)
        empty_graph_fixture.add_edge(bidirectional_edge)

        other_node = Node(id_="c")
        node4 = Node(id_="d")
        other_edge: Edge[Node] = Edge(src_node=other_node, dst_node=node4)
        other_graph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph(
            nodes=[other_node, node4], edges=[other_edge]
        )

        result = empty_graph_fixture.merge_multidigraph(other_graph)

        assert all(edge in result.edges for edge in [bidirectional_edge, other_edge])
        assert all(node in result.nodes for node in [node1, node2, other_node, node4])

    def test__merge_multidigraph__merges_nodes_and_edges(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test merging nodes and edges from another graph."""
        node1 = Node(id_="1")
        node2 = Node(id_="2")
        node3 = Node(id_="3")
        edge1: Edge[Node] = Edge(src_node=node1, dst_node=node2)
        empty_graph_fixture.add_node(node1)
        empty_graph_fixture.add_node(node2)
        empty_graph_fixture.add_edge(edge1)

        edge2: Edge[Node] = Edge(src_node=node2, dst_node=node3)
        other_graph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph(nodes=[node3], edges=[edge2])

        result = empty_graph_fixture.merge_multidigraph(other_graph)

        n_merged_nodes = 3
        n_merged_edges = 2
        assert len(result.nodes) == n_merged_nodes
        assert len(result.edges) == n_merged_edges

    def test__merge_multidigraph__raises_error_on_duplicate_nodes(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test that merging graphs with duplicate nodes raises ValueError."""
        node1 = Node(id_="1")
        empty_graph_fixture.add_node(node1)

        other_graph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph(nodes=[node1], edges=[])

        with pytest.raises(ValueError, match="Duplicate nodes found"):
            empty_graph_fixture.merge_multidigraph(other_graph)

    def test__remove_unconnected_components__keeps_largest_component(
        self, unstrongly_connected_multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Test removing unconnected components keeps the largest one."""
        initial_nodes_count = len(unstrongly_connected_multidigraph_fixture.nodes)
        components = unstrongly_connected_multidigraph_fixture.get_strongly_connected_components()
        largest_component_size = len(components[0])

        result = unstrongly_connected_multidigraph_fixture.remove_unconnected_components()

        assert len(result.nodes) == largest_component_size
        assert len(result.nodes) < initial_nodes_count
        assert result.get_is_strongly_connected() is True


class TestRepr:
    """The repr is a structural one-line summary, not a full field dump."""

    def test__repr__summarizes_structure(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """The summary reports counts, undirected dedup and connectivity."""
        summary = repr(multidigraph_fixture)

        assert summary.startswith("MultiDiGraph(")
        assert "nodes=6" in summary
        assert "edges_directed=13" in summary
        assert "edges_undirected=8" in summary
        assert "strongly_connected=False" in summary
        assert "components=2" in summary

    def test__repr__on_empty_graph_does_not_raise(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """The null graph summarizes as zero components instead of raising."""
        summary = repr(empty_graph_fixture)

        assert "nodes=0" in summary
        assert "components=0" in summary
        assert "strongly_connected=False" in summary

    def test__str__delegates_to_repr(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """`print(graph)` shows the summary, not the pydantic field dump."""
        assert str(multidigraph_fixture) == repr(multidigraph_fixture)


class TestAddBidirectionalEdgeClass:
    """The reversed copy keeps the concrete edge class (regression)."""

    def test__add_bidirectional_edge__preserves_edge_subclass(self) -> None:
        """A typed edge's reverse is the same class, not a base Edge."""
        a, b = Node(id_="a"), Node(id_="b")
        graph: MultiDiGraph[Node, WeightedEdge] = MultiDiGraph(nodes=[a, b], edges=[])

        graph.add_bidirectional_edge(WeightedEdge(src_node=a, dst_node=b, weight=2.0))

        assert all(isinstance(edge, WeightedEdge) for edge in graph.edges)
        assert all(edge.weight == pytest.approx(2.0) for edge in graph.edges)


class TestNodeIndex:
    """The private inverted index keeps node lookups O(1) and always in sync."""

    def test__filter_nodes_by__reads_from_the_index(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Filtering returns every node the index holds for that value."""
        node_a1 = Node(id_="1", category="A")
        node_a2 = Node(id_="3", category="A")
        empty_graph_fixture.add_node(node_a1)
        empty_graph_fixture.add_node(Node(id_="2", category="B"))
        empty_graph_fixture.add_node(node_a2)

        result = empty_graph_fixture.filter_nodes_by("category", "A")

        n_filtered = 2
        assert len(result) == n_filtered
        assert set(result) == {node_a1, node_a2}

    def test__filter_nodes_by__returns_a_copy_callers_cannot_corrupt(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Mutating the returned list must not affect the underlying index."""
        empty_graph_fixture.add_node(Node(id_="1", category="A"))

        result = empty_graph_fixture.filter_nodes_by("category", "A")
        result.clear()

        assert len(empty_graph_fixture.filter_nodes_by("category", "A")) == 1

    def test__filter_nodes_by__is_in_sync_after_add_node(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """A node added after construction is immediately filterable."""
        assert empty_graph_fixture.filter_nodes_by("category", "A") == []

        new_node = Node(id_="1", category="A")
        empty_graph_fixture.add_node(new_node)

        assert empty_graph_fixture.filter_nodes_by("category", "A") == [new_node]

    def test__filter_nodes_by__is_in_sync_after_remove_node(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """A removed node disappears from the index buckets."""
        node = Node(id_="1", category="A")
        empty_graph_fixture.add_node(node)
        assert empty_graph_fixture.filter_nodes_by("category", "A") == [node]

        empty_graph_fixture.remove_node(node)

        assert empty_graph_fixture.filter_nodes_by("category", "A") == []

    def test__filter_nodes_by__on_fixture_indexes_construction_nodes(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Nodes provided at construction time are indexed by the validator."""
        # feature_type_str is dumped, so it is a valid index key
        flights = multidigraph_fixture.filter_nodes_by("feature_type_str", "Flight")

        n_flights = 3
        assert len(flights) == n_flights

    def test__filter_nodes_by__unknown_attr_returns_empty(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """An attribute never seen on any node yields an empty result."""
        assert multidigraph_fixture.filter_nodes_by("does_not_exist", "whatever") == []

    def test__filter_nodes_by__unknown_value_returns_empty(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """A known attribute with an unseen value yields an empty result."""
        assert multidigraph_fixture.filter_nodes_by("feature_type_str", "Spaceship") == []

    def test__filter_nodes_by__indexes_non_string_values(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Float and bool attribute values are indexed by their raw value."""
        node = Node(id_="1", x=0.15, active=True)
        empty_graph_fixture.add_node(node)

        # Raw float and bool keys, not stringified
        assert empty_graph_fixture.filter_nodes_by("x", 0.15) == [node]
        assert empty_graph_fixture.filter_nodes_by("active", True) == [node]  # noqa: FBT003

    def test__merge_multidigraph__keeps_the_index_in_sync(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Nodes merged in from another graph become filterable and findable."""
        empty_graph_fixture.add_node(Node(id_="a", category="A"))
        merged_node = Node(id_="b", category="B")
        other_graph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph(nodes=[merged_node], edges=[])

        empty_graph_fixture.merge_multidigraph(other_graph)

        assert empty_graph_fixture.filter_nodes_by("category", "B") == [merged_node]
        assert empty_graph_fixture.get_node_by_id("b") == merged_node

    def test__relabel_nodes__rebuilds_the_index(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """After relabelling, old ids are gone and the new ids are indexed."""
        old_id = "f1"
        assert multidigraph_fixture.get_node_by_id(old_id).id_ == old_id

        multidigraph_fixture.relabel_nodes()

        with pytest.raises(ValueError, match=f"Node with id {old_id} not found"):
            multidigraph_fixture.get_node_by_id(old_id)

        for node in multidigraph_fixture.nodes:
            assert multidigraph_fixture.get_node_by_id(node.id_) is node

    def test__get_node_by_id__reads_from_the_index(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """A node is retrievable by id through the index bucket."""
        node = Node(id_="1")
        empty_graph_fixture.add_node(node)

        assert empty_graph_fixture.get_node_by_id("1") == node

    def test__get_node_by_id__raises_on_missing_id(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Looking up an unknown id raises ValueError."""
        with pytest.raises(ValueError, match="Node with id ghost not found"):
            empty_graph_fixture.get_node_by_id("ghost")


class TestSerialization:
    """The private index must never leak into the serialized model."""

    def test__model_dump__exposes_only_declared_fields(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """model_dump keys are exactly the declared fields (no index leak)."""
        dumped = multidigraph_fixture.model_dump()

        assert set(dumped.keys()) == {"nodes", "edges", "directed"}


class TestEdgeIndex:
    """The incidence index keeps ``edges_of``/``edges_between`` in sync with the edge list."""

    def test__edges_of__indexes_construction_edges(self) -> None:
        """Edges passed at construction are queryable from both endpoints."""
        node1, node2, node3 = Node(id_="1"), Node(id_="2"), Node(id_="3")
        edge12: Edge[Node] = Edge(src_node=node1, dst_node=node2)
        edge23: Edge[Node] = Edge(src_node=node2, dst_node=node3)
        graph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph(
            nodes=[node1, node2, node3], edges=[edge12, edge23]
        )

        assert graph.edges_of("1") == [edge12]
        assert graph.edges_of("2") == [edge12, edge23]
        assert graph.edges_of("3") == [edge23]

    def test__edges_of__is_in_sync_after_add_and_remove_edge(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """add_edge indexes the edge; remove_edge drops it from both endpoints."""
        node1, node2 = Node(id_="1"), Node(id_="2")
        edge: Edge[Node] = Edge(src_node=node1, dst_node=node2)

        empty_graph_fixture.add_edge(edge)
        assert empty_graph_fixture.edges_of("1") == [edge]
        assert empty_graph_fixture.edges_of("2") == [edge]

        empty_graph_fixture.remove_edge(edge)
        assert empty_graph_fixture.edges_of("1") == []
        assert empty_graph_fixture.edges_of("2") == []

    def test__edges_of__unknown_node_returns_empty(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """A node with no incident edge yields an empty list, not an error."""
        assert empty_graph_fixture.edges_of("ghost") == []

    def test__edges_of__returns_a_copy_callers_cannot_corrupt(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Mutating the returned list does not corrupt the index."""
        node1, node2 = Node(id_="1"), Node(id_="2")
        edge: Edge[Node] = Edge(src_node=node1, dst_node=node2)
        empty_graph_fixture.add_edge(edge)

        empty_graph_fixture.edges_of("1").clear()

        assert empty_graph_fixture.edges_of("1") == [edge]

    def test__edges_between__returns_both_directions_only_for_the_pair(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Both directions of the pair are returned; edges to other nodes are not."""
        node1, node2, node3 = Node(id_="1"), Node(id_="2"), Node(id_="3")
        forward: Edge[Node] = Edge(src_node=node1, dst_node=node2)
        backward: Edge[Node] = Edge(src_node=node2, dst_node=node1)
        other: Edge[Node] = Edge(src_node=node1, dst_node=node3)

        for edge in (forward, backward, other):
            empty_graph_fixture.add_edge(edge)

        # Insertion order, whichever endpoint the query starts from
        assert empty_graph_fixture.edges_between("1", "2") == [forward, backward]
        assert empty_graph_fixture.edges_between("2", "1") == [forward, backward]
        assert empty_graph_fixture.edges_between("2", "3") == []

    def test__add_bidirectional_edge__indexes_both_directions(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """The forward edge and its generated reverse are both indexed."""
        node1, node2 = Node(id_="1"), Node(id_="2")

        empty_graph_fixture.add_bidirectional_edge(Edge(src_node=node1, dst_node=node2))

        n_directions = 2
        assert len(empty_graph_fixture.edges_between("1", "2")) == n_directions

    def test__merge_multidigraph__keeps_the_edge_index_in_sync(
        self, empty_graph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """Edges brought in by a merge are queryable through the index."""
        node1, node2 = Node(id_="1"), Node(id_="2")
        edge: Edge[Node] = Edge(src_node=node1, dst_node=node2)
        other_graph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph(
            nodes=[node1, node2], edges=[edge]
        )

        empty_graph_fixture.merge_multidigraph(other_graph)

        assert empty_graph_fixture.edges_of("1") == [edge]

    def test__relabel_nodes__rebuilds_the_edge_index(
        self, multidigraph_fixture: MultiDiGraph[Node, Edge[Any]]
    ) -> None:
        """After relabelling, every edge is reachable from its new endpoint ids."""
        n_edges_before = len(multidigraph_fixture.edges)

        multidigraph_fixture.relabel_nodes()

        indexed = {
            id(edge)
            for node in multidigraph_fixture.nodes
            for edge in multidigraph_fixture.edges_of(node.id_)
        }
        assert len(multidigraph_fixture.edges) == n_edges_before
        assert indexed == {id(edge) for edge in multidigraph_fixture.edges}


class TestTraversal:
    """Test cases for the traversal queries."""

    def test__components_following__groups_the_undirected_projection(self) -> None:
        """With no policy, components are plain connectivity, isolated nodes alone."""
        graph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph()
        first, second, alone = Node(id_="a"), Node(id_="b"), Node(id_="c")

        for node in (first, second, alone):
            graph.add_node(node)

        graph.add_edge(Edge(src_node=first, dst_node=second))

        components = graph.components_following()

        assert {frozenset(c) for c in components} == {frozenset({"a", "b"}), frozenset({"c"})}

    def test__components_following__follows_only_accepted_edges(self) -> None:
        """The predicate is the traversal policy: a refused edge joins nothing."""
        graph: MultiDiGraph[Node, WeightedEdge] = MultiDiGraph()
        first, second, third = Node(id_="a"), Node(id_="b"), Node(id_="c")

        for node in (first, second, third):
            graph.add_node(node)

        graph.add_edge(WeightedEdge(src_node=first, dst_node=second, weight=1))
        graph.add_edge(WeightedEdge(src_node=second, dst_node=third, weight=9))

        components = graph.components_following(follow=lambda edge: edge.weight < 5)

        assert {frozenset(c) for c in components} == {frozenset({"a", "b"}), frozenset({"c"})}

    def test__components_following__never_enters_an_excluded_node(self) -> None:
        """What meets only through an excluded node stays apart."""
        graph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph()
        first, hub, second = Node(id_="a"), Node(id_="hub"), Node(id_="b")

        for node in (first, hub, second):
            graph.add_node(node)

        graph.add_edge(Edge(src_node=first, dst_node=hub))
        graph.add_edge(Edge(src_node=hub, dst_node=second))

        components = graph.components_following(exclude={"hub"})

        assert {frozenset(c) for c in components} == {frozenset({"a"}), frozenset({"b"})}

    def test__subgraph__matched_keeps_only_the_matches_and_their_edges(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """MATCHED never grows: the matched nodes, and the edges among them."""
        query = SubgraphQuery(node_ids=["a", "b"], expansion=Expansion.MATCHED)

        result = line_graph_fixture.filter_by(query)

        assert {node.id_ for node in result.nodes} == {"a", "b"}
        assert len(result.edges) == 1

    def test__subgraph__neighbours_stop_at_one_hop(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """NEIGHBOURS answers « what does this touch », never the hop after."""
        query = SubgraphQuery(node_ids=["a"], expansion=Expansion.NEIGHBOURS)

        result = line_graph_fixture.filter_by(query)

        assert {node.id_ for node in result.nodes} == {"a", "b"}

    def test__subgraph__component_grows_under_the_policy_plus_one_hop(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """COMPONENT expands along accepted edges; the refused frontier enters as a leaf."""
        result = line_graph_fixture.filter_by(
            SubgraphQuery(node_ids=["a"]), follow=lambda edge: edge.grade == "main"
        )

        # Component {a, b}; the branch to c is one hop of context; d stays out
        assert {node.id_ for node in result.nodes} == {"a", "b", "c"}
        assert len(result.edges) == 2

    def test__follow_edge_attrs__shape_the_expansion_the_way_a_callable_would(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """The query states the policy a caller would have passed in code, and it serializes."""
        result = line_graph_fixture.filter_by(
            SubgraphQuery(node_ids=["a"], follow_edge_attrs={"grade": ["main"]})
        )

        # Component {a, b}; the branch to c is one hop of context; d stays out
        assert {node.id_ for node in result.nodes} == {"a", "b", "c"}
        assert len(result.edges) == 2

    def test__several_values_on_one_attribute__are_ored(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """Any of the stated values lets the expansion through, so the whole line comes back."""
        result = line_graph_fixture.filter_by(
            SubgraphQuery(node_ids=["a"], follow_edge_attrs={"grade": ["main", "branch"]})
        )

        assert {node.id_ for node in result.nodes} == {"a", "b", "c", "d"}

    def test__a_stated_policy__narrows_the_callers_own_and_never_widens_it(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """Both must allow an edge: what the caller refuses, the query cannot re-open."""
        result = line_graph_fixture.filter_by(
            SubgraphQuery(node_ids=["a"], follow_edge_attrs={"grade": ["branch"]}),
            follow=lambda edge: edge.grade == "main",
        )

        # Nothing satisfies both, so a stands alone with one hop of context
        assert {node.id_ for node in result.nodes} == {"a", "b"}
        assert len(result.edges) == 1

    def test__an_edge_the_policy_refuses__is_still_returned_as_a_leaf(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """A policy bounds how far the answer GROWS, never what it shows at the frontier."""
        result = line_graph_fixture.filter_by(
            SubgraphQuery(node_ids=["a"], follow_edge_attrs={"grade": ["main"]})
        )

        # The refused branch b—c is in the answer; it was simply never walked through
        assert {(edge.src_node.id_, edge.dst_node.id_) for edge in result.edges} == {
            ("a", "b"),
            ("b", "c"),
        }

    @pytest.mark.parametrize("expansion", [Expansion.MATCHED, Expansion.NEIGHBOURS])
    @pytest.mark.parametrize("field", ["follow_edge_attrs", "avoid_edge_attrs"])
    def test__a_policy_on_an_expansion_that_never_traverses__is_refused(
        self, expansion: Expansion, field: str
    ) -> None:
        """Stating a policy no hop will ever consult is a mistake, not a silent no-op."""
        with pytest.raises(ValidationError, match="never walks through an edge"):
            SubgraphQuery(node_ids=["a"], expansion=expansion, **{field: {"grade": ["main"]}})

    def test__avoid_edge_attrs__refuse_the_hop_the_positive_form_would_have_to_enumerate(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """Naming the one kind to refuse says what listing every other kind would."""
        result = line_graph_fixture.filter_by(
            SubgraphQuery(node_ids=["a"], avoid_edge_attrs={"grade": ["branch"]})
        )

        # The same answer as following « main », stated the short way round
        assert {node.id_ for node in result.nodes} == {"a", "b", "c"}
        assert len(result.edges) == 2

    def test__a_value_named_on_both_sides__is_refused(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """The negative wins: admitting and refusing the same value is not a contradiction."""
        result = line_graph_fixture.filter_by(
            SubgraphQuery(
                node_ids=["a"],
                follow_edge_attrs={"grade": ["main", "branch"]},
                avoid_edge_attrs={"grade": ["branch"]},
            )
        )

        # « main » alone survives both, so the walk stops where the branch begins
        assert {node.id_ for node in result.nodes} == {"a", "b", "c"}

    def test__drop_edge_attrs__take_the_edge_out_of_the_answer_it_was_walked_through(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """Dropping prunes what came back; what lay beyond the dropped edge still came back."""
        result = line_graph_fixture.filter_by(
            SubgraphQuery(
                node_ids=["a"],
                follow_edge_attrs={"grade": ["main", "branch"]},
                drop_edge_attrs={"grade": ["branch"]},
            )
        )

        # d was reached THROUGH the branches, and stays though they are gone
        assert {node.id_ for node in result.nodes} == {"a", "b", "c", "d"}
        assert {edge.grade for edge in result.edges} == {"main"}

    def test__drop_node_attrs__take_the_node_out_without_detaching_what_lay_beyond(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """A dropped node was still walked through, so the far side of it survives."""
        result = line_graph_fixture.filter_by(
            SubgraphQuery(
                node_ids=["a"],
                follow_edge_attrs={"grade": ["main", "branch"]},
                drop_node_attrs={"region": ["south"]},
            )
        )

        # c and d are southern and go; a and b remain, and so does the island's absence
        assert {node.id_ for node in result.nodes} == {"a", "b"}
        assert len(result.edges) == 1  # only a—b has both endpoints left

    def test__subgraph__an_excluded_node_enters_as_a_leaf_never_traversed(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """An excluded node is attached where it touches, but joins nothing through it."""
        result = line_graph_fixture.filter_by(SubgraphQuery(node_ids=["a"]), exclude={"b"})

        assert {node.id_ for node in result.nodes} == {"a", "b"}
        assert len(result.edges) == 1

    def test__subgraph__criteria_combine_as_and(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """Each stated criterion narrows the match."""
        query = SubgraphQuery(
            node_attrs={"region": "north"}, id_contains="b", expansion=Expansion.MATCHED
        )

        result = line_graph_fixture.filter_by(query)

        # Region alone matches a, b and the island; the substring keeps b only
        assert {node.id_ for node in result.nodes} == {"b"}

    def test__subgraph__keep_node_attrs_prune_the_answer_not_the_expansion(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """The answer keeps only matching nodes; edges follow the survivors."""
        query = SubgraphQuery(node_ids=["a"], keep_node_attrs={"region": "north"})

        result = line_graph_fixture.filter_by(query)

        # The expansion reached c and d; the answer keeps the north alone
        assert {node.id_ for node in result.nodes} == {"a", "b"}
        assert len(result.edges) == 1

    def test__subgraph__edge_attrs_prune_the_answer_not_the_expansion(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """The edge filter thins what is shown, never how far the search reached."""
        query = SubgraphQuery(node_ids=["a"], edge_attrs={"grade": "branch"})

        result = line_graph_fixture.filter_by(query)

        assert {node.id_ for node in result.nodes} == {"a", "b", "c", "d"}
        assert {edge.src_node.id_ for edge in result.edges} == {"b", "c"}

    def test__subgraph__edge_bounds_drop_the_outside_and_the_unvalued(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """A window keeps the edges inside it; an unvalued edge cannot prove it is."""
        query = SubgraphQuery(node_ids=["a"], edge_bounds={"year": (2021, 2025)})

        result = line_graph_fixture.filter_by(query)

        # 2020 sits outside, the undated track cannot prove itself inside
        assert [edge.src_node.id_ for edge in result.edges] == ["b"]

    def test__subgraph__matching_nothing_returns_the_empty_graph(
        self, line_graph_fixture: MultiDiGraph[Node, Track]
    ) -> None:
        """No match is an empty answer, not an error."""
        result = line_graph_fixture.filter_by(SubgraphQuery(node_ids=["zzz"]))

        assert result.nodes == []
        assert result.edges == []

    def test__neighbours_of__returns_the_other_endpoint_of_each_edge(self) -> None:
        """Neighbourhood reads both edge directions through the index."""
        graph: MultiDiGraph[Node, Edge[Any]] = MultiDiGraph()
        centre, out, into = Node(id_="a"), Node(id_="b"), Node(id_="c")

        for node in (centre, out, into):
            graph.add_node(node)

        graph.add_edge(Edge(src_node=centre, dst_node=out))
        graph.add_edge(Edge(src_node=into, dst_node=centre))

        neighbours = graph.neighbours_of("a")

        assert {node.id_ for node in neighbours} == {"b", "c"}
