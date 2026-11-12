"""What a pre-check reports: an impossibility proven, a load warning, a search left open."""

from enum import StrEnum

from pydantic import BaseModel


class IssueKind(StrEnum):
    """How the impossibility was proven.

    Attributes:
        NO_PLACEMENT: An element has no placement on its own.
        GROUP: A small group of elements one hard constraint ties cannot be placed together.
        COMPONENT: Elements the hard constraints link, of any number, cannot be placed together.
        GROUP_LOAD: A fixed group sharing a resource demands more than its elements can use.
        POOL_LOAD: The elements that can only go to a pool of targets demand more than it offers.
        UNDECIDED: A group or component whose search ran out: nothing proven, nothing placed.
    """

    NO_PLACEMENT = "no_placement"
    GROUP = "group"
    COMPONENT = "component"
    GROUP_LOAD = "group_load"
    POOL_LOAD = "pool_load"
    UNDECIDED = "undecided"


class LoadGrade(StrEnum):
    """How a load compares with what it is offered.

    Attributes:
        OVERLOADED: The demand exceeds the offer: proven impossible.
        NEARLY_FULL: The demand nears the offer: possible, with little choice left.
        UNDERUSED: The demand is far below the offer of a pool of targets: possible.
    """

    OVERLOADED = "overloaded"
    NEARLY_FULL = "nearly_full"
    UNDERUSED = "underused"


class PrecheckIssue(BaseModel):
    """One impossibility (or a load warning), the elements involved, and the constraints used."""

    kind: IssueKind
    """How it was proven."""
    elements: list[int]
    """The element positions involved."""
    constraints: list[str]
    """The names of the hard constraints that suffice for the proof."""
    entity: str = ""
    """What it is about, by its name in the data: an element, a group's key, a pool's targets."""
    demand: int | None = None
    """For a load: what the elements demand (duration times demand with time)."""
    offer: int | None = None
    """For a load: the most the capacity can give them."""
    grade: LoadGrade | None = None
    """For a load: overloaded (a proof), nearly full or underused (warnings, no proof)."""

    @property
    def proven(self) -> bool:
        """Whether it proves an impossibility, rather than warning about a load or a search."""
        return self.kind != IssueKind.UNDECIDED and self.grade in {None, LoadGrade.OVERLOADED}
