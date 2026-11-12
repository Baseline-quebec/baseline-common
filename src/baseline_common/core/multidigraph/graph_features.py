"""Features for the multidigraph."""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class BaseGraphFeature(BaseModel):
    """Base class for graph features."""

    model_config = ConfigDict(extra="allow", frozen=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def feature_type_str(self) -> str:
        """The concrete subclass name (used to discriminate feature types)."""
        return self.__class__.__name__


class Node(BaseGraphFeature):
    """A node (or vertex) of a graph."""

    id_: str


class Edge[EndT: Node](BaseGraphFeature):
    """A directed edge between two nodes, typed by the node type it joins."""

    src_node: EndT
    dst_node: EndT


class Expansion(StrEnum):
    """How far a search grows around the nodes it matched."""

    COMPONENT = "component"
    """The component the traversal reaches, plus one hop of context."""

    NEIGHBOURS = "neighbours"
    """The matched nodes, plus one hop."""

    MATCHED = "matched"
    """The matched nodes alone, with the edges among them."""


class SubgraphQuery(BaseModel):
    """One search over a graph — serializable, so a tool call can carry it whole."""

    # Values within a field are ORed, criteria across fields ANDed; a pinned value is a scalar,
    # a set of alternatives a list, and a negative field beats its positive counterpart.

    model_config = ConfigDict(frozen=True)

    node_ids: list[str] = Field(
        default_factory=list, description="Exact node ids to match; any of them."
    )
    node_attrs: dict[str, Any] = Field(
        default_factory=dict,
        description="Attribute -> value every matched node must carry, read as "
        "``filter_nodes_by`` reads them.",
    )
    id_contains: str | None = Field(None, description="Substring the node id must contain.")
    expansion: Expansion = Field(
        Expansion.COMPONENT, description="How far the answer grows around the matched nodes."
    )
    keep_node_attrs: dict[str, Any] = Field(
        default_factory=dict,
        description="Attribute -> value a node must carry to stay in the answer; prunes "
        "the answer, never the expansion.",
    )
    edge_attrs: dict[str, Any] = Field(
        default_factory=dict,
        description="Attribute -> value an edge must carry to stay in the answer; prunes "
        "the answer, never the expansion.",
    )
    edge_bounds: dict[str, tuple[Any, Any]] = Field(
        default_factory=dict,
        description="Attribute -> inclusive (lower, upper) bounds an edge must sit inside, "
        "None leaving a side open; an edge missing the attribute falls outside.",
    )
    follow_edge_attrs: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Attribute -> the values that let the expansion walk THROUGH an edge; any "
        "of them, and every stated attribute must hold. This shapes how far the answer GROWS, "
        "where ``edge_attrs`` only prunes what already came back: an edge this rejects is still "
        "reached, and still returned, as a leaf. Empty leaves the graph's own policy in force.",
    )
    avoid_edge_attrs: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Attribute -> the values that stop the expansion from walking through an "
        "edge; carrying any one of them is enough. The negative of ``follow_edge_attrs`` and "
        "usually the one to reach for: « everything except the link to its document » names "
        "one kind, where the positive form would have to list every other kind.",
    )
    drop_node_attrs: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Attribute -> the values that keep a node OUT of the answer; carrying any "
        "one of them is enough. Prunes the answer, never the expansion — what is dropped was "
        "still walked through, so removing it never detaches what lay beyond it.",
    )
    drop_edge_attrs: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Attribute -> the values that keep an edge OUT of the answer; carrying any "
        "one of them is enough. Prunes the answer, never the expansion.",
    )

    @model_validator(mode="after")
    def validate_some_criterion(self) -> Self:
        """Refuse the query that matches by no criterion at all.

        Returns:
            The validated instance, unchanged.

        Raises:
            ValueError: If no matching criterion is stated.
        """
        if not self.node_ids and not self.node_attrs and self.id_contains is None:
            error_msg = "A query must state at least one matching criterion."
            raise ValueError(error_msg)

        return self

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Refuse an empty bounds window.

        Returns:
            The validated instance, unchanged.

        Raises:
            ValueError: If both bounds of an attribute are stated and cross.
        """
        for attr, (lower, upper) in self.edge_bounds.items():
            if lower is not None and upper is not None and upper < lower:
                error_msg = f"Empty window on {attr}: upper {upper} precedes lower {lower}."
                raise ValueError(error_msg)

        return self

    @model_validator(mode="after")
    def validate_follow_is_reachable(self) -> Self:
        """Refuse a traversal policy stated on a search that never traverses.

        Returns:
            The validated instance, unchanged.

        Raises:
            ValueError: If a follow policy is stated for an expansion that takes no hop.
        """
        if self.states_a_traversal_policy and self.expansion is not Expansion.COMPONENT:
            error_msg = (
                f"A follow policy needs '{Expansion.COMPONENT.value}' expansion; "
                f"'{self.expansion.value}' never walks through an edge."
            )
            raise ValueError(error_msg)

        return self

    def matches(self, node: Node) -> bool:
        """Whether a node is one the query asks about.

        Args:
            node: The node examined.

        Returns:
            True when every stated criterion holds.
        """
        if self.node_ids and node.id_ not in self.node_ids:
            return False

        if any(getattr(node, attr, None) != value for attr, value in self.node_attrs.items()):
            return False

        return self.id_contains is None or self.id_contains in node.id_

    def retains(self, node: Node) -> bool:
        """Whether the answer keeps a node the expansion reached.

        Args:
            node: The node examined.

        Returns:
            True when every kept-attribute matches and no dropped value is carried.
        """
        if self._refuses(node, self.drop_node_attrs):
            return False

        return all(
            getattr(node, attr, None) == value for attr, value in self.keep_node_attrs.items()
        )

    @property
    def states_a_traversal_policy(self) -> bool:
        """Whether the query shapes the expansion itself instead of leaving it to the graph."""
        return bool(self.follow_edge_attrs or self.avoid_edge_attrs)

    def follows(self, edge: Edge[Any]) -> bool:
        """Whether the expansion may walk through an edge, rather than merely reach it.

        Args:
            edge: The edge examined.

        Returns:
            True when every admitted attribute holds and no refused one does.
        """
        return self._admits(edge, self.follow_edge_attrs) and not self._refuses(
            edge, self.avoid_edge_attrs
        )

    @staticmethod
    def _admits(feature: Node | Edge[Any], stated: dict[str, list[Any]]) -> bool:
        """Whether a feature carries one of the stated values for every stated attribute.

        Args:
            feature: The node or edge examined.
            stated: Attribute -> the values that admit it; empty admits everything.

        Returns:
            True when every stated attribute holds.
        """
        # A StrEnum attribute compares equal to its own printing, so a query that arrived as
        # JSON states its values in plain strings and still matches the typed feature.
        return all(getattr(feature, attr, None) in values for attr, values in stated.items())

    @staticmethod
    def _refuses(feature: Node | Edge[Any], stated: dict[str, list[Any]]) -> bool:
        """Whether a feature carries any one of the stated values, on any stated attribute.

        Args:
            feature: The node or edge examined.
            stated: Attribute -> the values that refuse it; empty refuses nothing.

        Returns:
            True as soon as one stated attribute holds.
        """
        return any(getattr(feature, attr, None) in values for attr, values in stated.items())

    def keeps(self, edge: Edge[Any]) -> bool:
        """Whether an edge of the answer survives the query's filters.

        Args:
            edge: The edge examined.

        Returns:
            True when the attributes match, nothing dropped is carried and every bound holds.
        """
        if self._refuses(edge, self.drop_edge_attrs):
            return False

        if any(getattr(edge, attr, None) != value for attr, value in self.edge_attrs.items()):
            return False

        for attr, (lower, upper) in self.edge_bounds.items():
            current = getattr(edge, attr, None)

            if current is None:
                return False

            if lower is not None and current < lower:
                return False

            if upper is not None and current > upper:
                return False

        return True
