"""One pre-check run: what the hard constraints tie in a bound model, and each element's domain."""

import itertools
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from baseline_core.core.optimization.problem_formulation.declaration import (
    AssignedGroup,
    FilteredSet,
    IndexSet,
    Member,
    Position,
    Size,
    SortedList,
    TargetRead,
    ValueRead,
)
from baseline_core.core.optimization.problem_formulation.expression import Expr
from baseline_core.core.optimization.problem_formulation.model import Model
from baseline_core.core.optimization.solver.binding.binder import BoundData
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.kernel.kernel import Candidates, Filters, LoneDomain
from baseline_core.core.optimization.solver.kernel.shape import Template, TermKind
from baseline_core.core.optimization.solver.stop_condition import StopCondition


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from baseline_core.core.optimization.solver.kernel.kernel import Kernel


@dataclass(frozen=True, slots=True)
class Link:
    """A hard instance that ties elements together."""

    constraint: int
    """Its constraint, as a name position."""
    elements: list[int]
    """The elements it ties, sorted."""


@dataclass(frozen=True, slots=True)
class InstanceReads:
    """A hard instance that reads elements besides its free members."""

    constraint: int
    """Its constraint, as a name position."""
    members: set[int]
    """Its free members."""
    reads: set[int]
    """The elements it reads besides them."""


@dataclass(frozen=True, slots=True)
class HardTies:
    """What ties the elements: the links, and the instances that read beyond their members."""

    links: list[Link]
    """Per hard instance tying elements, the elements it ties."""
    instances: list[InstanceReads]
    """Per hard instance that reads besides its free members, what it reads."""


@dataclass(frozen=True, slots=True)
class PrecheckRun:
    """What one pre-check run reads: the bound model, its hard structure, each lone domain."""

    kernel: "Kernel"
    """The bound kernel, searched with every other element absent."""
    model: Model[Any]
    """The model bound."""
    data: BoundData
    """The problem's data as the model reads it: the declarations' values, the targets."""
    elements: IndexSet[Any]
    """The set the decisions are over."""
    stops: list[StopCondition]
    """What ends the run, asked with iteration 0; none, and every search runs to its cap."""
    hard: list[int]
    """The hard constraints, as name positions."""
    on_start: list[int]
    """The hard constraints that read the integer decision alone: they filter starts."""
    on_set: list[int]
    """The hard constraints that read the assignment alone: filters of an element's sets."""
    span: list[int]
    """Per element, its span (0 without one)."""
    among: list[list[int]] | None
    """Per element, the sets it may take; ``None``: every set."""
    links: list[Link]
    """Per hard instance tying elements: its constraint and the elements it ties."""
    neighbors: list[set[int]]
    """Per element, the elements some hard instance ties it to."""
    needs: list[dict[int, set[int] | None]]
    """Per element, per constraint: the elements present before it is read around it."""
    set_level: set[int]
    """The constraints with an instance per decided set (read whole, set by set)."""
    takers: set[int]
    """The elements some decided set may hold."""
    starts: list[list[int]]
    """Per element whose domain is known, the starts level 1 leaves it."""
    sets: list[list[int]]
    """Per element whose domain is known, the sets level 1 leaves it."""
    placeable: list[bool]
    """Per element whose domain is known, whether level 1 leaves it a start and a set."""
    filtered: "NDArray[np.int64]"
    """Per element and constraint, how many placements level 1 removed."""
    arrays: "dict[tuple[int, ...], NDArray[np.int64]]"
    """Lists of constraints as the kernel reads them, made once each."""
    fill: dict[tuple[int, frozenset[int]], int]
    """The most a run of slots holds of pieces of given lengths, worked out once each."""

    @property
    def names(self) -> list[str]:
        """The constraint names, in the kernel's count order."""
        return self.model.names

    @property
    def known(self) -> int:
        """How many elements, from the first, have their lone domain read."""
        return len(self.starts)

    @property
    def late(self) -> bool:
        """Whether one of the run's stops says to give up."""
        return any(stop.should_stop(0) for stop in self.stops)

    def lone(self, element: int) -> LoneDomain:
        """The domain level 1 leaves an element, with what each constraint removed from it.

        Args:
            element: An element whose domain is known.

        Returns:
            Its starts, its sets and the blame.
        """
        return LoneDomain(self.starts[element], self.sets[element], self.filtered[element])

    @classmethod
    def read(cls, bound: BoundModel, stops: list[StopCondition]) -> "PrecheckRun":
        """What ties the elements, then each one's placements alone, until a stop says to give up.

        Args:
            bound: The bound model; its kernel ends as it was given.
            stops: What ends the run: nothing more is read once one says to give up.

        Returns:
            The run.
        """
        with cls.isolated(bound.kernel):
            return cls._with_domains(cls._start(bound, stops))

    @staticmethod
    @contextmanager
    def isolated(kernel: "Kernel") -> Generator[None]:
        """Every element absent inside the block, the kernel given back as it was after.

        Args:
            kernel: The bound kernel.

        Yields:
            Nothing: the kernel may change inside.
        """
        starts, targets = kernel.placement
        absent = kernel.absent_mask

        try:
            kernel.set_absent(np.ones(kernel.n, dtype=np.int64))
            yield
        finally:
            kernel.set_absent(absent)
            kernel.set_placement(starts, targets)

    @classmethod
    def _start(cls, bound: BoundModel, stops: list[StopCondition]) -> "PrecheckRun":
        """Read what ties the elements, no domain read yet.

        Args:
            bound: The bound model.
            stops: What ends the run: nothing more is read once one says to give up.

        Returns:
            The run.
        """
        kernel = bound.kernel
        model, templates, data = bound.model, bound.templates, bound.data
        interval, assign = model.integer_decision, model.assignment

        hard = [index for index, is_hard in enumerate(model.hard) if is_hard]

        rows = (
            None
            if assign is None or assign.among is None
            else [sorted(set(row)) for row in data.subsets[assign.among]]
        )

        holders: list[list[int]] = [[] for _ in range(kernel.n_sets)]

        for element in range(kernel.n):
            for target in rows[element] if rows is not None else range(kernel.n_sets):
                holders[target].append(element)

        ties = cls._read_ties(templates, hard, holders)
        return cls(
            kernel=kernel,
            model=model,
            data=data,
            elements=cast("IndexSet[Any]", model.decided_set),
            stops=stops,
            hard=hard,
            on_start=cls._filters(model, hard, "value"),
            on_set=cls._filters(model, hard, "target"),
            span=(
                [0] * kernel.n
                if interval is None or interval.span is None
                else [int(value) for value in data.values[interval.span]]
            ),
            among=rows,
            links=ties.links,
            neighbors=cls._neighbors(ties.links, kernel.n),
            needs=cls._read_needs(ties, kernel.n),
            set_level={template.name_index for template in templates if template.set_level},
            takers={element for held in holders for element in held},
            starts=[],
            sets=[],
            placeable=[],
            filtered=np.zeros((kernel.n, len(model.names)), dtype=np.int64),
            arrays={},
            fill={},
        )

    @classmethod
    def _with_domains(cls, run: "PrecheckRun") -> "PrecheckRun":
        """The run with each element's lone domain, read in order until the run's stop.

        Args:
            run: The run, no domain read yet.

        Returns:
            The run with its domains.
        """
        starts: list[list[int]] = []
        sets: list[list[int]] = []

        for element in range(run.kernel.n):
            if run.late:
                break

            domain = run.domain(element, run.hard)
            run.filtered[element] += domain.blame
            starts.append(domain.starts)
            sets.append(domain.sets)

        placeable = [bool(lone) and bool(held) for lone, held in zip(starts, sets, strict=True)]
        return replace(run, starts=starts, sets=sets, placeable=placeable)

    def domain(self, element: int, constraint_indices: list[int]) -> LoneDomain:
        """The starts and sets the element may take alone, each filtered by the constraints on it.

        Args:
            element: The element.
            constraint_indices: The constraints that may filter, as name positions.

        Returns:
            A superset of its true placements: start 0 without time, set 0 without an assignment.
        """
        # a constraint reading other elements around this one cannot be read with them absent
        alone = {index for index in constraint_indices if self.ready(element, index, {element})}

        interval = self.model.integer_decision

        if interval is None:
            low, high = 0, 1
        else:
            low = interval.low
            high = interval.high - self.span[element] + 1

        targets = self.among[element] if self.among is not None else range(self.kernel.n_sets)
        candidates = Candidates(low, high, np.asarray(targets, dtype=np.int64))
        filters = Filters(
            self.array([index for index in self.on_start if index in alone]),
            self.array([index for index in self.on_set if index in alone]),
        )

        domain: LoneDomain = self.kernel.lone_placements(element, candidates, filters)
        return domain

    def array(self, constraint_indices: list[int]) -> "NDArray[np.int64]":
        """The constraints as the kernel reads them, made once per list.

        Args:
            constraint_indices: The constraints, as name positions.

        Returns:
            The array.
        """
        key = tuple(constraint_indices)

        if key not in self.arrays:
            self.arrays[key] = np.asarray(constraint_indices, dtype=np.int64)

        return self.arrays[key]

    def ready(self, element: int, constraint_index: int, present: set[int]) -> bool:
        """Whether a constraint is read exactly around an element once ``present`` are placed.

        Args:
            element: The element.
            constraint_index: The constraint, as a name position.
            present: The elements placed.

        Returns:
            Every element its instances around ``element`` read is present.
        """
        needed = self.needs[element].get(constraint_index, set())
        return needed is not None and needed <= present

    def linked_groups(self, limit: int) -> list[list[int]]:
        """Per hard constraint, the elements its instances tie together, in small groups.

        Args:
            limit: The most elements a group holds.

        Returns:
            The groups of more than one and at most ``limit`` elements, smallest first.
        """
        by_constraint: dict[int, list[list[int]]] = {index: [] for index in self.hard}

        for link in self.links:
            if len(link.elements) <= limit:
                by_constraint[link.constraint].append(link.elements)

        groups: set[tuple[int, ...]] = set()

        for links in by_constraint.values():
            groups |= {
                tuple(part) for part in self._parts(links, self.kernel.n) if len(part) <= limit
            }

        return sorted((list(group) for group in groups), key=len)

    def components(self) -> list[list[int]]:
        """The elements every hard constraint links, transitively: one group per component.

        Returns:
            The components of more than one element, smallest first.
        """
        return sorted(self._parts([link.elements for link in self.links], self.kernel.n), key=len)

    @staticmethod
    def _neighbors(links: list[Link], n: int) -> list[set[int]]:
        """Per element, the elements some hard instance ties it to.

        Args:
            links: Per hard instance tying elements, the elements it ties.
            n: The number of elements.

        Returns:
            The neighbors, each element among its own when it is tied at all.
        """
        neighbors: list[set[int]] = [set() for _ in range(n)]

        for link in links:
            for element in link.elements:
                neighbors[element].update(link.elements)

        return neighbors

    @classmethod
    def _filters(cls, model: Model[Any], hard: list[int], decision: str) -> list[int]:
        """The hard constraints that read one decision alone: they filter it element by element.

        Args:
            model: The model.
            hard: The hard constraints, as name positions.
            decision: ``value`` (the integer decision) or ``target`` (the assignment).

        Returns:
            The hard constraints that read that decision and nothing else.
        """
        index_of = {name: position for position, name in enumerate(model.names)}
        reads_of: dict[int, set[str]] = {}

        for term in model.terms:
            if index_of[term.name] in hard:
                reads_of.setdefault(index_of[term.name], set()).update(cls._reads(term.expr))

        return [index for index in hard if reads_of.get(index) == {decision}]

    @classmethod
    def _read_ties(
        cls, templates: list[Template], hard: list[int], holders: list[list[int]]
    ) -> HardTies:
        """Read, per hard instance, the elements it ties and those it reads besides its own.

        Args:
            templates: The templates the kernel was generated from.
            hard: The hard constraints, as name positions.
            holders: Per set, the elements that may be assigned to it.

        Returns:
            The links, and the instances that read besides their free members.
        """
        instances: list[InstanceReads] = []
        links: list[Link] = []

        for template in templates:
            if not template.shape.reads_decision or template.name_index not in hard:
                continue

            free = template.shape.kind in {TermKind.MEMBER, TermKind.PAIR}
            pair = template.shape.kind == TermKind.PAIR
            members_of = cls._per_instance(template.members, template.member_offsets)
            reads_of = cls._per_instance(template.reads, template.read_offsets)

            for members, reads in zip(members_of, reads_of, strict=True):
                # a decided set (negative) ties nothing; the elements it may hold count as read
                tied = {entry for entry in reads if entry >= 0}

                if pair or (free and tied):
                    tied.update(entry for entry in members if entry >= 0)

                if len(tied) > 1:
                    links.append(Link(template.name_index, sorted(tied)))

                if reads:
                    instances.append(
                        InstanceReads(
                            template.name_index,
                            cls._held(members, holders) if free else set(),
                            cls._held(reads, holders),
                        )
                    )

        return HardTies(links, instances)

    @classmethod
    def _read_needs(cls, ties: HardTies, n: int) -> list[dict[int, set[int] | None]]:
        """Per element and constraint, the elements to place before reading it around the element.

        Args:
            ties: What ties the elements, and what each hard instance reads.
            n: The number of elements.

        Returns:
            The needs; ``None`` where a read is larger than any linked part.
        """
        # reads larger than any linked part never fit in a group search: never read there
        parts = cls._parts([link.elements for link in ties.links], n)
        largest = max((len(part) for part in parts), default=1)
        needs: list[dict[int, set[int] | None]] = [{} for _ in range(n)]

        for instance in ties.instances:
            needed = instance.reads if len(instance.reads) <= largest else None

            for element in instance.members | instance.reads:
                known = needs[element].get(instance.constraint, set())
                needs[element][instance.constraint] = (
                    None if known is None or needed is None else known | needed
                )

        return needs

    @staticmethod
    def _per_instance(flat: "NDArray[np.int64]", offsets: "NDArray[np.int64]") -> list[list[int]]:
        """Split a template's flat per-instance lists.

        Args:
            flat: The entries of every instance, one after another.
            offsets: Where each instance's entries start, plus the end.

        Returns:
            One list per instance.
        """
        entries, bounds = flat.tolist(), offsets.tolist()
        return [entries[low:high] for low, high in itertools.pairwise(bounds)]

    @staticmethod
    def _held(entries: list[int], holders: list[list[int]]) -> set[int]:
        """The elements of an instance's entries, a decided set ``-1 - r`` for those it may hold.

        Args:
            entries: The entries: elements, or ``-1 - r`` for every element of set ``r``.
            holders: Per set, the elements that may be assigned to it.

        Returns:
            The elements.
        """
        out = {entry for entry in entries if entry >= 0}

        for entry in entries:
            if entry < 0:
                out.update(holders[-1 - entry])

        return out

    @staticmethod
    def _reads(expr: Expr) -> set[str]:
        """Which decisions an expression reads.

        Args:
            expr: The expression.

        Returns:
            ``value`` (the integer decision) and/or ``target`` (the assignment).
        """
        out: set[str] = set()
        todo: list[Expr] = [expr]

        while todo:
            node = todo.pop()

            if isinstance(node, ValueRead):
                out.add("value")

            if isinstance(node, TargetRead):
                out.add("target")

            if isinstance(node, Member | Size | Position):
                collection = node.sorted_list if isinstance(node, Position) else node.collection

                while isinstance(collection, FilteredSet | SortedList):
                    collection = collection.base

                if isinstance(collection, AssignedGroup):
                    out.add("target")

            todo += node.children

        return out

    @classmethod
    def _parts(cls, links: list[list[int]], n: int) -> list[list[int]]:
        """The connected parts the links make among the elements.

        Args:
            links: Elements tied together, one list per tie.
            n: The number of elements.

        Returns:
            Each part of more than one element, sorted.
        """
        parent = list(range(n))

        for refs in links:
            for element in refs[1:]:
                parent[cls._root(parent, element)] = cls._root(parent, refs[0])

        parts: dict[int, list[int]] = {}

        for element in range(n):
            parts.setdefault(cls._root(parent, element), []).append(element)

        return [part for part in parts.values() if len(part) > 1]

    @staticmethod
    def _root(parent: list[int], element: int) -> int:
        """The representative of an element's part, halving the path on the way.

        Args:
            parent: Per element, its parent in the union-find forest.
            element: The element.

        Returns:
            The root of its tree.
        """
        while parent[element] != element:
            parent[element] = parent[parent[element]]
            element = parent[element]

        return element
