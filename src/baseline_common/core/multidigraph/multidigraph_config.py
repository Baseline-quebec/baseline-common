"""Schema of a typed graph: which node classes it holds and the relations between them."""

from pydantic import field_validator

from baseline_common.core.multidigraph.graph_features import Edge, Node
from baseline_common.core.multidigraph.multidigraph import MultiDiGraph


class NodeConfig(Node):
    """One node type of the schema: the class of its nodes (e.g. ``Customer``)."""

    node_class: type[Node]


class EdgeConfig(Edge[NodeConfig]):
    """One relation of the schema, from one node type to another (e.g. ``purchased``)."""

    relation: str


class MultiDiGraphConfig(MultiDiGraph[NodeConfig, EdgeConfig]):
    """The schema of a typed graph, itself a graph of node types joined by relations."""

    @field_validator("nodes")
    @classmethod
    def validate_unique_node_classes(cls, v: list[NodeConfig]) -> list[NodeConfig]:
        """Validate that no node class is described by two node types.

        Args:
            v: Node types to validate.

        Returns:
            The node types, unchanged.

        Raises:
            ValueError: If two node types share a node class.
        """
        node_classes = [node.node_class for node in v]

        if len(node_classes) != len(set(node_classes)):
            error_msg = "Node classes must be unique."
            raise ValueError(error_msg)

        return v

    def get_node_config(self, node_class: type[Node]) -> NodeConfig:
        """Find the node type describing a node class.

        Args:
            node_class: The class of the nodes.

        Returns:
            The node type whose ``node_class`` is ``node_class``.

        Raises:
            ValueError: If no node type describes ``node_class``.
        """
        node_config = next((node for node in self.nodes if node.node_class is node_class), None)

        if node_config is None:
            error_msg = f"No node config for the node class {node_class.__name__}."
            raise ValueError(error_msg)

        return node_config

    def get_edge_config(self, src_class: type[Node], dst_class: type[Node]) -> EdgeConfig:
        """Find the relation joining two node classes, in that direction.

        Args:
            src_class: The class of the source nodes.
            dst_class: The class of the destination nodes.

        Returns:
            The relation from ``src_class`` to ``dst_class``.

        Raises:
            ValueError: If no relation joins the two classes in that direction.
        """
        # Unambiguous: the graph holds one edge per (source, destination) pair.
        edge_config = next(
            (
                edge
                for edge in self.edges
                if edge.src_node.node_class is src_class and edge.dst_node.node_class is dst_class
            ),
            None,
        )

        if edge_config is None:
            error_msg = f"No edge config from {src_class.__name__} to {dst_class.__name__}."
            raise ValueError(error_msg)

        return edge_config
