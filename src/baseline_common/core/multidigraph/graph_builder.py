"""Build a typed graph from records, checked against its schema."""

from typing import Any

from baseline_common.core.multidigraph.graph_features import Edge, Node
from baseline_common.core.multidigraph.multidigraph import MultiDiGraph
from baseline_common.core.multidigraph.multidigraph_config import MultiDiGraphConfig


class GraphBuilder:
    """Turns node and edge records into a graph whose every element the schema describes."""

    def __init__(self, config: MultiDiGraphConfig) -> None:
        """Bind the schema the graphs are built to.

        Args:
            config: The schema every node and edge must follow.
        """
        self.config = config

    def build[NodeT: Node, EdgeT: Edge[Any]](
        self, nodes: list[NodeT], edges: list[EdgeT]
    ) -> MultiDiGraph[NodeT, EdgeT]:
        """Build a new graph from records, each edge labelled with its schema relation.

        Args:
            nodes: The node records, of any class the schema describes.
            edges: The edge records, joining nodes given in ``nodes``.

        Returns:
            The graph of ``nodes`` and ``edges``, edges joining the graph's own node instances.
        """
        for node in nodes:
            self.config.get_node_config(type(node))

        graph: MultiDiGraph[NodeT, EdgeT] = MultiDiGraph(nodes=list(nodes))

        for edge in edges:
            src_node = graph.get_node_by_id(edge.src_node.id_)
            dst_node = graph.get_node_by_id(edge.dst_node.id_)
            relation = self.config.get_edge_config(type(src_node), type(dst_node)).relation
            update = {"src_node": src_node, "dst_node": dst_node, "relation": relation}
            graph.add_edge(edge.model_copy(update=update))

        return graph
