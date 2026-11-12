"""Convert a typed graph to PyTorch Geometric ``HeteroData``, the input of heterogeneous GNNs."""

from typing import Any

import torch
from torch_geometric.data import HeteroData

from baseline_core.core.ml.graph.graph_feature_config import Feature, GraphFeatureConfig
from baseline_core.core.multidigraph.graph_features import Node
from baseline_core.core.multidigraph.multidigraph import MultiDiGraph
from baseline_core.core.multidigraph.multidigraph_config import MultiDiGraphConfig


class HeteroDataConverter:
    """Turns a graph built against a schema into feature matrices and edge indexes per type."""

    def __init__(
        self, graph_config: MultiDiGraphConfig, feature_config: GraphFeatureConfig
    ) -> None:
        """Initialize the converter.

        Args:
            graph_config: The schema naming the node types and the relations.
            feature_config: The features read from each node type of the schema.

        Raises:
            ValueError: If the features are not given for exactly the schema's node types.
        """
        node_types = sorted(node.id_ for node in graph_config.nodes)

        if sorted(feature_config.features) != node_types:
            error_msg = f"Features must be given for exactly the node types {node_types}."
            raise ValueError(error_msg)

        self.graph_config = graph_config
        self.feature_config = feature_config

    def to_hetero_data(self, graph: MultiDiGraph[Any, Any]) -> HeteroData:
        """Convert a graph: one feature matrix per node type, one edge index per relation.

        Args:
            graph: The graph, each node and edge described by the schema.

        Returns:
            Types keyed by the schema's ``id_``; one without nodes or edges is left out.
        """
        rows: dict[str, list[list[float]]] = {}
        row_of: dict[str, int] = {}

        for node in graph.nodes:
            node_type = self.graph_config.get_node_config(type(node)).id_
            type_rows = rows.setdefault(node_type, [])
            row_of[node.id_] = len(type_rows)
            type_rows.append(self.feature_vector(node, self.feature_config.features[node_type]))

        edge_indexes: dict[tuple[str, str, str], list[list[int]]] = {}

        for edge in graph.edges:
            edge_config = self.graph_config.get_edge_config(
                type(edge.src_node), type(edge.dst_node)
            )
            edge_type = (edge_config.src_node.id_, edge_config.relation, edge_config.dst_node.id_)
            sources, targets = edge_indexes.setdefault(edge_type, [[], []])
            sources.append(row_of[edge.src_node.id_])
            targets.append(row_of[edge.dst_node.id_])

        data = HeteroData()

        for node_type, type_rows in rows.items():
            data[node_type].x = torch.tensor(type_rows, dtype=torch.float)
            data[node_type].num_nodes = len(type_rows)

        for edge_type, edge_index in edge_indexes.items():
            data[edge_type].edge_index = torch.tensor(edge_index, dtype=torch.long)

        return data

    @staticmethod
    def feature_vector(node: Node, features: list[Feature]) -> list[float]:
        """Read a node's features, transformed, as one flat vector.

        Args:
            node: The node read.
            features: The features to read, in order.

        Returns:
            The features in order, a list or array feature flattened in place.
        """
        vector: list[float] = []

        for feature in features:
            value = getattr(node, feature.name)

            if feature.transform is not None:
                value = feature.transform(value)

            vector.extend(torch.as_tensor(value, dtype=torch.float).flatten().tolist())

        return vector
