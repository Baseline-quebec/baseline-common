"""Data model of the MultiDiGraph."""

import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, Literal, Self

import networkx as nx
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from baseline_core.core.multidigraph.graph_features import Edge, Expansion, Node, SubgraphQuery


class MultiDiGraph[NodeT: Node, EdgeT: Edge[Any]](BaseModel):
    """Directed multi-edge graph, generic over its node and edge types."""

    # The indexes hold only while membership changes go through ``add_*``/``remove_*`` (or a
    # reassignment then ``_rebuild_index``); mutating ``nodes``/``edges`` directly bypasses them.

    # Fixed-order categorical hues for ``plot`` (CVD-validated); overflow folds, never cycles.
    PLOT_PALETTE: ClassVar[tuple[str, ...]] = (
        "#2a78d6",  # blue
        "#eb6834",  # orange
        "#1baf7a",  # aqua
        "#eda100",  # yellow
        "#e87ba4",  # magenta
        "#008300",  # green
        "#4a3aa7",  # violet
        "#e34948",  # red
    )

    nodes: list[NodeT] = Field(default_factory=list)
    edges: list[EdgeT] = Field(default_factory=list)
    directed: bool = True

    _node_index: dict[str, dict[Any, list[NodeT]]] = PrivateAttr(default_factory=dict)
    _edge_index: dict[str, list[EdgeT]] = PrivateAttr(default_factory=dict)

    @field_validator("nodes")
    @classmethod
    def validate_unique_nodes(cls, v: list[NodeT]) -> list[NodeT]:
        """Validate that every node id is unique.

        Args:
            v: Nodes to validate.

        Returns:
            The validated node list, unchanged when uniqueness holds.

        Raises:
            ValueError: If two nodes share the same ``id_``.
        """
        node_types = [node.id_ for node in v]

        if len(node_types) != len(set(node_types)):
            error_msg = "Node types must be unique."
            raise ValueError(error_msg)

        return v

    @field_validator("edges")
    @classmethod
    def validate_unique_edges(cls, v: list[EdgeT]) -> list[EdgeT]:
        """Validate that every edge relation is unique.

        Args:
            v: Edges to validate.

        Returns:
            The validated edge list, unchanged when uniqueness holds.

        Raises:
            ValueError: If two edges share the same ``(src_node, dst_node)`` pair.
        """
        edge_relations = [(edge.src_node.id_, edge.dst_node.id_) for edge in v]

        if len(edge_relations) != len(set(edge_relations)):
            edge_relations_str = [
                f"{src_node} -> {dst_node}" for src_node, dst_node in edge_relations
            ]
            error_msg = f"Edge relations must be unique. {', '.join(edge_relations_str)}"
            raise ValueError(error_msg)

        return v

    @model_validator(mode="after")
    def validate_networkx_graph(self) -> Self:
        """Validate that the wrapped graph can be projected to networkx.

        Returns:
            The validated instance, unchanged when the projection succeeds.

        Raises:
            ValueError: If projecting to ``nx.MultiDiGraph`` fails.
        """
        try:
            _ = self.networkx_graph
        except Exception as e:
            error_msg = f"Error creating networkx graph: {e}"
            raise ValueError(error_msg) from e

        return self

    @model_validator(mode="after")
    def _build_index(self) -> Self:
        """Populate the node and edge indexes from the validated fields.

        Returns:
            The validated instance, both indexes built.
        """
        self._rebuild_index()
        return self

    @staticmethod
    def _index_node(node: NodeT, index: dict[str, dict[Any, list[NodeT]]]) -> None:
        """Add a single node to the given inverted index.

        Args:
            node: The node to index.
            index: The inverted index mutated in place.
        """
        for attr, value in node.model_dump().items():
            if isinstance(value, (list, dict, set)):
                continue

            index.setdefault(attr, {}).setdefault(value, []).append(node)

    def _deindex_node(self, node: NodeT) -> None:
        """Remove a single node from every bucket its dumped attrs point to.

        Args:
            node: The node to remove from the index.
        """
        for attr, value in node.model_dump().items():
            if isinstance(value, (list, dict, set)):
                continue

            bucket = self._node_index.get(attr, {}).get(value)

            if bucket is None or node not in bucket:
                continue

            bucket.remove(node)

            if not bucket:
                del self._node_index[attr][value]

                if not self._node_index[attr]:
                    del self._node_index[attr]

    def _index_edge(self, edge: EdgeT) -> None:
        """Add a single edge to the incidence index, under both endpoints.

        Args:
            edge: The edge to index.
        """
        for node_id in {edge.src_node.id_, edge.dst_node.id_}:
            self._edge_index.setdefault(node_id, []).append(edge)

    def _deindex_edge(self, edge: EdgeT) -> None:
        """Remove a single edge from both endpoint buckets, dropping empty ones.

        Args:
            edge: The edge to remove from the index.
        """
        for node_id in {edge.src_node.id_, edge.dst_node.id_}:
            bucket = self._edge_index.get(node_id)

            if bucket is None or edge not in bucket:
                continue

            bucket.remove(edge)

            if not bucket:
                del self._edge_index[node_id]

    def _rebuild_index(self) -> None:
        """Clear both indexes and re-index every current node and edge."""
        self._node_index.clear()

        for node in self.nodes:
            self._index_node(node, self._node_index)

        self._edge_index.clear()

        for edge in self.edges:
            self._index_edge(edge)

    def __repr__(self) -> str:
        """Summarize the graph structure instead of dumping every node and edge.

        Returns:
            A one-line summary of the counts and strong connectivity.
        """
        node_types = Counter(node.feature_type_str for node in self.nodes)
        components = self.get_strongly_connected_components()
        undirected = len({frozenset((e.src_node.id_, e.dst_node.id_)) for e in self.edges})
        return (
            f"{type(self).__name__}("
            f"nodes={len(self.nodes)}, "
            f"node_types={dict(sorted(node_types.items()))}, "
            f"edges_directed={len(self.edges)}, "
            f"edges_undirected={undirected}, "
            f"strongly_connected={len(components) == 1}, "
            f"components={len(components)})"
        )

    def __str__(self) -> str:
        """Print the structural summary (see :meth:`__repr__`).

        Returns:
            The ``repr`` of the graph.
        """
        return repr(self)

    @property
    def networkx_graph(self) -> "nx.MultiDiGraph":
        """This graph as a networkx ``MultiDiGraph``: same nodes, edges and ``directed`` flag."""
        graph: nx.MultiDiGraph = nx.MultiDiGraph()

        graph.graph["directed"] = self.directed

        for node in self.nodes:
            graph.add_node(node.id_, **node.model_dump(exclude={"id_"}))

        for edge in self.edges:
            graph.add_edge(
                edge.src_node.id_,
                edge.dst_node.id_,
                **edge.model_dump(exclude={"src_node", "dst_node"}),
            )

        return graph

    def add_node(self, node: NodeT) -> str:
        """Add a node to the MultiDiGraph.

        Args:
            node: The node to add.

        Returns:
            The id of the newly appended node.
        """
        self.nodes.append(node)
        self._index_node(node, self._node_index)
        return node.id_

    def remove_node(self, node: NodeT) -> None:
        """Remove a node from the MultiDiGraph.

        Args:
            node: The node to remove.
        """
        self.nodes.remove(node)
        self._deindex_node(node)

    def add_edge(self, edge: EdgeT) -> None:
        """Add an edge to the MultiDiGraph.

        Args:
            edge: The edge to add.
        """
        self.edges.append(edge)
        self._index_edge(edge)

    def remove_edge(self, edge: EdgeT) -> None:
        """Remove an edge from the MultiDiGraph.

        Args:
            edge: The edge to remove.
        """
        self.edges.remove(edge)
        self._deindex_edge(edge)

    def edges_of(self, node_id: str) -> list[EdgeT]:
        """Edges incident to a node (as source or destination), via the index.

        Args:
            node_id: Identifier of the node whose incident edges are wanted.

        Returns:
            The incident edges, as a fresh copy of the index bucket.
        """
        return list(self._edge_index.get(node_id, []))

    def edges_between(self, node_id_a: str, node_id_b: str) -> list[EdgeT]:
        """Edges linking two nodes in either direction, via the index.

        Args:
            node_id_a: Identifier of one endpoint.
            node_id_b: Identifier of the other endpoint.

        Returns:
            The edges between the two nodes, both directions.
        """
        pair = {node_id_a, node_id_b}
        return [
            edge
            for edge in self._edge_index.get(node_id_a, [])
            if {edge.src_node.id_, edge.dst_node.id_} == pair
        ]

    def get_node_by_id(self, node_id: str) -> NodeT:
        """Look up a node by its identifier.

        Args:
            node_id: Identifier of the node to find.

        Returns:
            The node registered under ``node_id``.

        Raises:
            ValueError: If no node is registered under ``node_id``.
        """
        nodes = self._node_index.get("id_", {}).get(node_id)

        if not nodes:
            error_msg = f"Node with id {node_id} not found."
            raise ValueError(error_msg)

        return nodes[0]

    def group_nodes_by(self, attr: str) -> dict[Any, list[NodeT]]:
        """Group the nodes by the given attribute using the inverted index.

        Args:
            attr: Attribute name to read from each node's ``model_dump``.

        Returns:
            Mapping ``attribute value -> nodes``, as a shallow copy.

        Raises:
            ValueError: If NO indexed node carries ``attr`` — not merely one lacking it.
        """
        if attr not in self._node_index:
            error_msg = f"Node type attribute {attr} is not found in the node."
            raise ValueError(error_msg)

        return dict(self._node_index[attr])

    def filter_nodes_by(self, attr: str, value: Any) -> list[NodeT]:  # noqa: ANN401
        """Filter the nodes by the given attribute and value via the index.

        Args:
            attr: Attribute name to read from each node's ``model_dump``.
            value: Expected value the attribute must match.

        Returns:
            The matching nodes, as a fresh copy of the index bucket.
        """
        return list(self._node_index.get(attr, {}).get(value, []))

    @staticmethod
    def _traverses(
        query: SubgraphQuery, follow: Callable[[EdgeT], bool] | None, edge: EdgeT
    ) -> bool:
        """Whether an expansion may walk through one edge, under both policies at once.

        Args:
            query: The search, carrying the policy a serialized caller stated.
            follow: The caller's own policy; every edge when None.
            edge: The edge examined.

        Returns:
            True when both allow it — the query narrows the caller's policy, never widens it.
        """
        return query.follows(edge) and (follow is None or follow(edge))

    def filter_by(
        self,
        query: SubgraphQuery,
        follow: Callable[[EdgeT], bool] | None = None,
        exclude: set[str] | None = None,
    ) -> Self:
        """Search the graph: the subgraph the query asks for, not just the matching nodes.

        Args:
            query: What to match, how far to grow, which edges survive.
            follow: Which edges a COMPONENT expansion may cross, ANDed with the query's own.
            exclude: Node ids the expansion never enters; they can still enter as leaves.

        Returns:
            A new graph of the kept nodes and surviving edges; empty when nothing matches.
        """
        seeds = {node.id_ for node in self.nodes if query.matches(node)}
        kept = set(seeds)

        if query.expansion is Expansion.COMPONENT:
            policy = partial(self._traverses, query, follow)

            for component in self.components_following(policy, exclude):
                if component & seeds:
                    kept |= component

        if query.expansion is not Expansion.MATCHED:
            for node_id in list(kept):
                for edge in self.edges_of(node_id):
                    kept.add(edge.src_node.id_)
                    kept.add(edge.dst_node.id_)

        nodes = [node for node in self.nodes if node.id_ in kept and query.retains(node)]
        ids = {node.id_ for node in nodes}
        return type(self)(
            nodes=nodes,
            edges=[
                edge
                for edge in self.edges
                if edge.src_node.id_ in ids and edge.dst_node.id_ in ids and query.keeps(edge)
            ],
        )

    def add_bidirectional_edge(self, edge: EdgeT) -> None:
        """Add an edge and its reverse to the graph.

        Args:
            edge: The forward edge; a reversed copy is added alongside it.
        """
        self.add_edge(edge)
        reversed_edge = type(edge)(
            src_node=edge.dst_node,
            dst_node=edge.src_node,
            **edge.model_dump(exclude={"src_node", "dst_node"}),
        )

        self.add_edge(reversed_edge)

    def merge_multidigraph(self, other: Self) -> Self:
        """Merge another MultiDiGraph into this one in place.

        Args:
            other: Graph whose nodes and edges are appended to ``self``.

        Returns:
            ``self`` after the merge, for chaining.

        Raises:
            ValueError: If any node id is present in both graphs.
        """
        ids = {node.id_ for node in self.nodes}
        other_ids = {node.id_ for node in other.nodes}
        duplicate_ids = ids.intersection(other_ids)

        if duplicate_ids:
            error_msg = (
                f"Duplicate nodes found. The nodes of the graphs must be unique. "
                f"Duplicate ids: {duplicate_ids}"
            )
            raise ValueError(error_msg)

        for node in other.nodes:
            self.add_node(node)

        for edge in other.edges:
            self.add_edge(edge)

        return self

    def get_strongly_connected_components(self) -> list[list[str]]:
        """Compute strongly connected components, largest first.

        Returns:
            Components as lists of node ids, sorted by descending size.
        """
        strongly_connected_components = [
            list(component) for component in nx.strongly_connected_components(self.networkx_graph)
        ]
        sorted_strongly_components = sorted(strongly_connected_components, key=len, reverse=True)
        return sorted_strongly_components

    def get_is_strongly_connected(self) -> bool:
        """Check if the graph is strongly connected.

        Returns:
            ``True`` if the graph is strongly connected.
        """
        is_strongly_connected: bool = nx.is_strongly_connected(self.networkx_graph)
        return is_strongly_connected

    def relabel_nodes(self) -> Self:
        """Relabel every node in place with a fresh UUID.

        Returns:
            The MultiDiGraph with the relabeled nodes (``self``).
        """
        networkx_graph = self.networkx_graph
        mapping = {old_node: str(uuid.uuid4()) for old_node in networkx_graph.nodes()}
        nx.relabel_nodes(networkx_graph, mapping, copy=False)

        new_multigraph = self.from_networkx_graph(networkx_graph)
        self.nodes = new_multigraph.nodes
        self.edges = new_multigraph.edges
        self._rebuild_index()

        return new_multigraph

    def remove_unconnected_components(self) -> Self:
        """Keep only the largest strongly connected component, dropping the rest.

        Returns:
            The MultiDiGraph restricted to the largest component (``self``).

        Raises:
            ValueError: If the resulting graph is not strongly connected.
        """
        components = self.get_strongly_connected_components()
        largest_component = max(components, key=len)

        networkx_graph = self.networkx_graph
        subgraph = networkx_graph.subgraph(largest_component)

        new_multigraph = self.from_networkx_graph(subgraph)
        self.nodes = new_multigraph.nodes
        self.edges = new_multigraph.edges
        self._rebuild_index()

        if not new_multigraph.get_is_strongly_connected():
            error_msg = (
                f"The graph is still not strongly connected after removing unconnected "
                f"components ({len(components)} components)."
            )
            raise ValueError(error_msg)

        return new_multigraph

    @staticmethod
    def get_strongly_connected_components_data(
        connected_components: list[list[str]],
    ) -> tuple[list[set[str]], list[list[str]], list[list[int]]]:
        """Project components into sets, id lists, and per-component index lists.

        Args:
            connected_components: Components as lists of node ids.

        Returns:
            ``(components, node_id_lists, node_index_lists)``, indices contiguous across all.
        """
        components_set = [set(component) for component in connected_components]
        components_node_ids = [list(component) for component in connected_components]

        components_node_idx = []
        start = 0

        for component in components_node_ids:
            components_node_idx.append(list(range(start, start + len(component))))
            start += len(component)

        return components_set, components_node_ids, components_node_idx

    def plot(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
        self,
        figsize: tuple[int, int] = (12, 8),
        node_size: int = 3000,
        node_color: str = "lightblue",
        font_size: int = 10,
        edge_color: str = "gray",
        with_labels: bool = True,  # noqa: FBT001, FBT002
        show_node_features: bool = False,  # noqa: FBT001, FBT002
        edge_label_attr: str | None = None,
        node_color_attr: str | None = None,
        edge_color_attr: str | None = None,
        layout: Literal[
            "spring", "circular", "kamada_kawai", "planar", "random", "shell", "spectral"
        ] = "spring",
        save_path: str | None = None,
        highlight: dict[Any, str] | None = None,
    ) -> None:
        """Plot the graph with its features.

        Args:
            figsize: Size of the matplotlib figure.
            node_size: Size of the rendered nodes.
            node_color: Fill color for the nodes.
            font_size: Font size used for labels.
            edge_color: Color used for the edges.
            with_labels: Whether to render node labels.
            show_node_features: Whether to include node attributes in the labels.
            edge_label_attr: Attribute name to render as edge labels, if any.
            node_color_attr: Attribute to color the nodes by; without it, ``node_color``.
            edge_color_attr: Attribute to color the edges by; without it, ``edge_color``.
            layout: networkx layout algorithm name.
            save_path: Path to save the figure to; if ``None`` the plot is shown.
            highlight: An edge value's hue on every figure, others in ``edge_color``; None: all.
        """
        import matplotlib.pyplot as plt

        graph = self.networkx_graph

        plt.figure(figsize=figsize)

        pos: Any

        if layout == "spring":
            pos = nx.spring_layout(graph, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(graph, scale=2)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph, scale=2)
        elif layout == "planar":
            pos = nx.planar_layout(graph, scale=2)
        elif layout == "random":
            pos = nx.random_layout(graph, seed=42)
        elif layout == "shell":
            pos = nx.shell_layout(graph, scale=2)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph, scale=2)

        # One fixed hue per attribute value; the fallback color for the rest
        node_values = [
            data.get(node_color_attr) if node_color_attr else None
            for _, data in graph.nodes(data=True)
        ]
        node_hues = self._plot_colors(node_values, node_color)
        edge_values = [
            data.get(edge_color_attr) if edge_color_attr else None
            for *_, data in graph.edges(data=True)
        ]

        if highlight is not None:
            # A hue on every kind reads as noise; only the edges asked for stand out of the gray,
            # each in the hue it was given, so one kind reads the same across a series of figures.
            edge_values = [value if value in highlight else None for value in edge_values]
            edge_hues: dict[Any, str] = {None: edge_color, **highlight}
        else:
            edge_hues = self._plot_colors(edge_values, edge_color)

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_size,
            node_color=[node_hues[value] for value in node_values],
            alpha=0.9,
        )

        # A highlighted edge is drawn bold and opaque: a hue alone is lost under its neighbours.
        standing_out = [highlight is not None and value is not None for value in edge_values]
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color=[edge_hues[value] for value in edge_values],
            alpha=[1.0 if bold else 0.6 for bold in standing_out],
            arrows=True,
            arrowsize=20,
            arrowstyle="->",
            width=[5.0 if bold else 2.0 for bold in standing_out],
            node_size=node_size,
        )

        if with_labels:  # noqa: PLR1702
            if show_node_features:
                node_labels = {}

                for node in self.nodes:
                    all_attrs = node.model_dump(exclude_none=True, exclude_unset=True)

                    meta_attrs = {"id_", "linked_to", "feature_type_str"}

                    feature_items = []

                    for k, v in all_attrs.items():
                        if k not in meta_attrs:
                            if k == "features" and isinstance(v, list):
                                feature_names: list[str] = []

                                for feat in v:
                                    if isinstance(feat, dict) and "name" in feat:
                                        name = feat.get("alias") or feat.get("name")

                                        if name is not None:
                                            feature_names.append(str(name))

                                if feature_names:
                                    feature_items.append(f"features: {', '.join(feature_names)}")

                            elif not isinstance(v, (list, dict)) or (
                                isinstance(v, list) and v and not isinstance(v[0], dict)
                            ):
                                feature_items.append(f"{k}: {v}")

                    if feature_items:
                        feature_str = "\n".join(feature_items)
                        node_labels[node.id_] = f"{node.id_}\n{feature_str}"
                    else:
                        node_labels[node.id_] = node.id_

                nx.draw_networkx_labels(
                    graph,
                    pos,
                    labels=node_labels,
                    font_size=font_size - 2,
                    font_color="black",
                    verticalalignment="center",
                    horizontalalignment="center",
                    bbox={
                        "boxstyle": "round,pad=0.5",
                        "facecolor": "white",
                        "edgecolor": "gray",
                        "alpha": 0.95,
                        "linewidth": 1.5,
                    },
                )
            else:
                node_labels = {node.id_: node.id_ for node in self.nodes}
                nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=font_size)

        if edge_label_attr:
            edge_labels = {}

            for u, v, data in graph.edges(data=True):
                if edge_label_attr in data:
                    edge_labels[u, v] = data[edge_label_attr]
                else:
                    edge_attrs = {k: v for k, v in data.items() if k != "feature_type_str"}

                    if edge_attrs:
                        edge_labels[u, v] = "\n".join([f"{k}: {v}" for k, v in edge_attrs.items()])

            if edge_labels:
                nx.draw_networkx_edge_labels(
                    graph, pos, edge_labels=edge_labels, font_size=font_size, font_color="black"
                )

        # A node label runs far wider than its node and is clipped to the axes; the margin is
        # what keeps « address:civic|1500||recollets|trois-rivieres » from losing its end.
        plt.margins(0.12)
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    @classmethod
    def _plot_colors(cls, values: list[Any], fallback: str) -> dict[Any, str]:
        """One fixed hue per distinct value, assigned in sorted order.

        Args:
            values: One attribute value per drawn element, None when absent.
            fallback: Color for None and for values past the palette.

        Returns:
            Mapping ``value -> color``, covering every given value.
        """
        present = sorted({value for value in values if value is not None}, key=str)
        hues: dict[Any, str] = {None: fallback}

        for index, value in enumerate(present):
            hues[value] = cls.PLOT_PALETTE[index] if index < len(cls.PLOT_PALETTE) else fallback

        return hues

    def neighbours_of(self, node_id: str) -> list[Node]:
        """The other endpoint of every edge incident to a node, via the index.

        Args:
            node_id: Identifier of the node whose neighbourhood is read.

        Returns:
            The neighbouring nodes, one entry per incident edge (base ``Node``, not ``NodeT``).
        """
        return [
            edge.dst_node if edge.src_node.id_ == node_id else edge.src_node
            for edge in self.edges_of(node_id)
        ]

    def components_following(
        self, follow: Callable[[EdgeT], bool] | None = None, exclude: set[str] | None = None
    ) -> list[set[str]]:
        """Connected components of the undirected projection, under a traversal policy.

        Args:
            follow: Which edges the traversal may follow; every edge when None.
            exclude: Node ids the traversal never enters.

        Returns:
            The components, in the order their first node was added.
        """
        barred = exclude or set()
        links: dict[str, set[str]] = defaultdict(set)

        for edge in self.edges:
            if follow is not None and not follow(edge):
                continue

            source, target = edge.src_node.id_, edge.dst_node.id_

            if source in barred or target in barred:
                continue

            links[source].add(target)
            links[target].add(source)

        components: list[set[str]] = []
        seen: set[str] = set()

        for node in self.nodes:
            if node.id_ in seen or node.id_ in barred:
                continue

            component, stack = set(), [node.id_]
            seen.add(node.id_)

            while stack:
                current = stack.pop()
                component.add(current)

                for neighbour in links[current]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)

            components.append(component)

        return components

    @classmethod
    def from_networkx_graph(cls, graph: "nx.MultiDiGraph") -> Self:
        """Create a MultiDiGraph from a networkx ``MultiDiGraph``.

        Args:
            graph: Source networkx graph to convert.

        Returns:
            A MultiDiGraph mirroring the source graph.
        """
        nodes: list[Any] = [Node(id_=str(node), **data) for node, data in graph.nodes(data=True)]

        node_map = {node.id_: node for node in nodes}

        edges: list[Any] = [
            Edge(src_node=node_map[str(source)], dst_node=node_map[str(target)], **data)
            for source, target, data in graph.edges(data=True)
        ]

        return cls(nodes=nodes, edges=edges)
