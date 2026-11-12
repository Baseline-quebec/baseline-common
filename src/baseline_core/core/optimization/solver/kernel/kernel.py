# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=False, initializedcheck=False
"""The evaluator a model compiles to: placement, scores and searches around generated counts."""

# Cython pure Python mode: `KernelSource` appends the model's generated subclass to this source.
# cdivision=False keeps Python's `//` (rounded down), compiled or not.
# Cython's typed shadow cannot express memoryviews (`cython.long[::1]`), so `cython` stays
# untyped and its decorators (`cfunc`, `exceptval`, ...) with it.
# mypy: disable-error-code="untyped-decorator"
# pyright: reportIndexIssue=false, reportInvalidTypeArguments=false, reportAssignmentType=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportReturnType=false
# pyright: reportArgumentType=false

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cython
import numpy as np
from numpy.typing import ArrayLike, NDArray


if TYPE_CHECKING:
    from baseline_core.core.optimization.solver.stop_condition import StopCondition


@dataclass(frozen=True)
class Candidates:
    """The placements an element may try alone: a range of starts and the targets."""

    low: int
    """The first start tried."""
    high: int
    """The start after the last one tried."""
    targets: NDArray[np.int64]
    """The sets tried."""


@dataclass(frozen=True)
class Filters:
    """The constraints that filter an element's starts, and those that filter its sets."""

    on_start: NDArray[np.int64]
    """Name positions of the constraints read on its starts."""
    on_set: NDArray[np.int64]
    """Name positions of the constraints read on its sets."""


@dataclass(frozen=True)
class LoneDomain:
    """What an element may take alone, every other element absent."""

    starts: list[int]
    """The starts with no violation alone."""
    sets: list[int]
    """The sets with no violation alone."""
    blame: NDArray[np.int64]
    """Per constraint, how many starts or sets it removed."""


@dataclass(frozen=True)
class SearchPlan:
    """A group to place alone together: its order, its candidates, what each placement reads."""

    order: NDArray[np.int64]
    """The elements, in the order they are placed."""
    domains: NDArray[np.int64]
    """Per element, its starts then its sets: its candidates are their product."""
    domain_offsets: NDArray[np.int64]
    """Where each element's starts, then sets, begin, plus the end."""
    checks: NDArray[np.int64]
    """Per element, what its placement reads, then, per element, what is read forward."""
    check_offsets: NDArray[np.int64]
    """Where each of these ``2 * len(order)`` lists begins, plus the end."""
    whole: NDArray[np.int64]
    """The constraints read once every element is placed."""
    closed: NDArray[np.int64]
    """The constraints the group closes: all read once every element is present."""


@dataclass(frozen=True)
class Domains:
    """Per element, the starts then the sets it may take: its placements are their product."""

    values: NDArray[np.int64]
    """Per element, its starts then its sets, one element after another."""
    offsets: NDArray[np.int64]
    """Where each element's starts, then sets, begin, plus the end: ``2 * n + 1`` entries."""


@dataclass(frozen=True)
class SearchOutcome:
    """What a search of a group found."""

    found: int
    """1 placeable, 0 proven not, -1 undecided."""
    work: int
    """The evaluations spent."""
    blame: NDArray[np.int64]
    """Per constraint, how many candidates it rejected."""

    @property
    def verdict(self) -> bool | None:
        """``True`` placeable, ``False`` proven not, ``None`` undecided."""
        return None if self.found < 0 else bool(self.found)

    @property
    def blamed(self) -> set[int]:
        """The constraints that rejected something on the way, as name positions."""
        return set(np.flatnonzero(self.blame).tolist())


@dataclass(frozen=True)
class KernelLayout:
    """What a kernel counts and how it reports: elements, sets, constraints and their names."""

    n: int
    """Number of elements."""
    n_sets: int
    """Number of sets elements can be assigned to."""
    names: list[str]
    """The constraint names, in count order."""
    variables: list[str | None]
    """``[integer variable, assignment]`` names, ``None`` where the model has none."""
    labels: list[list[str]]
    """Per generated template, the label of each of its instances."""
    objects: list[object]
    """The schema object each element stands for, to fill a label's ``{a}``, ``{b}``."""
    setlevel: ArrayLike
    """Per constraint, 1 when its instances are whole decided sets."""
    weights: ArrayLike
    """Per constraint, its weight."""
    hard: ArrayLike
    """Per constraint, whether it is hard."""


@cython.cclass
class Kernel:
    """A model over a data set: elements placed at a value in a target, counted per constraint."""

    # Bare assignment: an annotation on a non-literal is ignored by the cclass transform.
    COMPILED = cython.compiled
    """True when this module runs as a compiled extension rather than pure Python."""

    n = cython.declare(cython.long, visibility="readonly")
    n_sets = cython.declare(cython.long, visibility="readonly")
    n_names = cython.declare(cython.long, visibility="readonly")
    soft_mode = cython.declare(cython.bint, visibility="public")
    names = cython.declare(list, visibility="readonly")
    variables = cython.declare(list, visibility="readonly")
    labels = cython.declare(list, visibility="readonly")
    objects = cython.declare(list, visibility="readonly")
    sid: cython.long
    start: cython.long[::1]
    target: cython.long[::1]
    absent: cython.long[::1]
    head: cython.long[::1]
    nxt: cython.long[::1]
    prv: cython.long[::1]
    size: cython.long[::1]
    buf: cython.long[::1]
    scope: cython.long[::1]
    elements: cython.long[::1]
    others: cython.long[::1]
    w: cython.long[::1]
    hard: cython.long[::1]
    act: cython.long[::1]
    before: cython.long[::1]
    cnt: cython.long[::1]
    change: cython.long[::1]
    offset: cython.long[::1]
    setlevel: cython.long[::1]

    def __init__(self, layout: KernelLayout) -> None:
        """Allocate the placement and per-constraint tables, every element present at 0 in set 0.

        Args:
            layout: The elements, sets and constraints counted, and the names reported.
        """
        n: cython.long = layout.n
        n_sets: cython.long = layout.n_sets
        self.n, self.n_sets, self.n_names = n, n_sets, len(layout.names)
        self.names, self.variables = layout.names, layout.variables
        self.labels, self.objects = layout.labels, layout.objects
        self.sid, self.soft_mode = 0, False

        size = max(n, 1)
        self.start = np.zeros(size, dtype=np.int64)
        self.target = np.zeros(size, dtype=np.int64)
        self.absent = np.zeros(size, dtype=np.int64)

        self.head = np.full(max(n_sets, 1), -1, dtype=np.int64)
        self.nxt = np.full(size, -1, dtype=np.int64)
        self.prv = np.full(size, -1, dtype=np.int64)
        self.size = np.zeros(max(n_sets, 1), dtype=np.int64)

        self.buf = np.zeros(size, dtype=np.int64)
        self.scope = np.zeros(size, dtype=np.int64)
        self.elements = np.zeros(size, dtype=np.int64)
        self.others = np.full(size, -1, dtype=np.int64)

        self.w = np.ascontiguousarray(layout.weights, dtype=np.int64)
        self.hard = np.ascontiguousarray(layout.hard, dtype=np.int64)
        self.act = np.zeros(self.n_names, dtype=np.int64)
        self.before = np.zeros(self.n_names, dtype=np.int64)
        self.cnt = np.zeros(self.n_names, dtype=np.int64)
        self.change = np.zeros(2, dtype=np.int64)
        self.offset = np.zeros(self.n_names, dtype=np.int64)
        self.setlevel = np.ascontiguousarray(layout.setlevel, dtype=np.int64)

        self.set_placement(np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))

    @cython.cfunc
    @cython.exceptval(check=False)
    def _around(self, j: cython.long, r_other: cython.long) -> cython.void:
        """Add to ``cnt`` the ``act`` instances element ``j`` can change (generated).

        Args:
            j: The element.
            r_other: Another set whose decided instances it can change (its old or new set), or -1.
        """

    @cython.cfunc
    @cython.exceptval(check=False)
    def _full(self) -> cython.void:
        """Add to ``cnt`` every instance of the ``act`` constraints (generated)."""

    @cython.cfunc
    @cython.exceptval(check=False)
    def _around_set(self, n_elements: cython.long) -> cython.void:
        """Like ``_around``, over the first ``n_elements`` of ``elements`` and ``others``, each once.

        Args:
            n_elements: How many of ``elements`` are in the set.
        """

    @cython.cfunc
    @cython.exceptval(check=False)
    def _constants(self) -> cython.void:
        """Add to ``cnt`` every instance that reads no decision (generated)."""

    def _count_constants(self) -> None:
        """Keep per constraint the count of instances reading no decision: they never change."""
        i: cython.long
        self._zero(self.cnt)
        self._constants()

        for i in range(self.n_names):
            self.offset[i] = self.cnt[i]

    def _violations(self) -> list[list[Any]]:  # noqa: PLR6301  the generated override reads self
        """``[template, instance, members, constraint index]`` per violation (generated)."""
        return []

    @cython.cfunc
    @cython.exceptval(check=False)
    def _link(self, j: cython.long, r: cython.long) -> cython.void:
        """Push a present element onto its set's list.

        Args:
            j: The element.
            r: Its set.
        """
        h: cython.long

        if self.absent[j] != 0:
            return

        h = self.head[r]
        self.nxt[j] = h
        self.prv[j] = -1

        if h >= 0:
            self.prv[h] = j

        self.head[r] = j
        self.size[r] += 1

    @cython.cfunc
    @cython.exceptval(check=False)
    def _unlink(self, j: cython.long, r: cython.long) -> cython.void:
        """Take a present element off its set's list.

        Args:
            j: The element.
            r: Its set.
        """
        p: cython.long
        q: cython.long

        if self.absent[j] != 0:
            return

        p, q = self.prv[j], self.nxt[j]

        if p >= 0:
            self.nxt[p] = q
        else:
            self.head[r] = q

        if q >= 0:
            self.prv[q] = p

        self.size[r] -= 1

    @cython.cfunc
    @cython.exceptval(check=False)
    def _place(self, j: cython.long, v: cython.long, r: cython.long) -> cython.void:
        """Move an element to value ``v`` in set ``r``.

        Args:
            j: The element.
            v: Its new value.
            r: Its new set.
        """
        self._unlink(j, self.target[j])
        self.start[j] = v
        self.target[j] = r
        self._link(j, r)

    def set_placement(self, starts: ArrayLike, targets: ArrayLike) -> None:
        """Load a placement: one start and one target set per element.

        Args:
            starts: Per element, its value.
            targets: Per element, its set.

        Raises:
            ValueError: The arrays do not match the elements, or a set is out of range.
        """
        element: cython.long
        set_index: cython.long
        start_of: cython.long[::1] = np.ascontiguousarray(starts, dtype=np.int64)
        target_of: cython.long[::1] = np.ascontiguousarray(targets, dtype=np.int64)

        if start_of.shape[0] != self.n or target_of.shape[0] != self.n:
            msg = "placement arrays must match the kernel's element count"
            raise ValueError(msg)

        for element in range(self.n):
            if target_of[element] < 0 or target_of[element] >= self.n_sets:
                msg = (
                    f"set {target_of[element]} of element {element} is outside [0, {self.n_sets})"
                )
                raise ValueError(msg)

        for set_index in range(self.n_sets):
            self.head[set_index] = -1
            self.size[set_index] = 0

        for element in range(self.n):
            self.start[element] = start_of[element]
            self.target[element] = target_of[element]
            self._link(element, target_of[element])

    def load(self, **arrays: ArrayLike) -> None:
        """Load a whole schedule: one array per variable, named as in the model.

        Args:
            arrays: The schedule, by variable name.

        Raises:
            ValueError: The names are not the model's variables.
        """
        value_name, set_name = self.variables
        wanted = sorted(name for name in self.variables if name is not None)

        if sorted(arrays) != wanted:
            msg = f"the model's variables are {wanted}; got {sorted(arrays)}"
            raise ValueError(msg)

        zeros = np.zeros(self.n, dtype=np.int64)
        self.set_placement(
            arrays[value_name] if value_name else zeros, arrays[set_name] if set_name else zeros
        )

    @property
    def placement(self) -> list[list[int]]:
        """The ``[starts, targets]`` of every element, as plain lists."""
        starts, targets = np.asarray(self.start)[: self.n], np.asarray(self.target)[: self.n]
        return [starts.tolist(), targets.tolist()]

    @property
    def absent_mask(self) -> list[int]:
        """The absent flags, one per element: what ``set_absent`` takes."""
        flags: list[int] = np.asarray(self.absent)[: self.n].tolist()
        return flags

    @property
    def violations(self) -> list[list[object]]:
        """The violated instances of the current placement: ``[name, label, members]`` each."""
        out: list[list[object]] = []

        for template, instance, members, name_index in self._violations():
            involved = {
                "abcdefgh"[slot]: self.objects[member] for slot, member in enumerate(members)
            }
            label = self.labels[template][instance].format(**involved)
            out.append([self.names[name_index], label, members])

        return out

    @cython.cfunc
    @cython.exceptval(-1)
    def _check(self, element: cython.long, target: cython.long) -> cython.int:
        """Refuse an element or a set out of range.

        Args:
            element: The element.
            target: The set.

        Returns:
            0 when both are in range.

        Raises:
            ValueError: One of them is out of range.
        """
        if element < 0 or element >= self.n or target < 0 or target >= self.n_sets:
            msg = f"element {element} or set {target} out of range"
            raise ValueError(msg)

        return 0

    @cython.ccall
    def insert(self, element: cython.long, value: cython.long, target: cython.long) -> None:
        """Make an element present, at a value in a set.

        Args:
            element: The element.
            value: Its value.
            target: Its set.
        """
        self._check(element, target)

        if self.absent[element] != 0:
            self.absent[element] = 0
            self.start[element] = value
            self.target[element] = target
            self._link(element, target)
        else:
            self._place(element, value, target)

    @cython.ccall
    def remove(self, element: cython.long) -> None:
        """Leave an element out: it counts in no constraint, as if it did not exist.

        Args:
            element: The element.
        """
        self._check(element, 0)

        if self.absent[element] == 0:
            self._unlink(element, self.target[element])
            self.absent[element] = 1

    def set_absent(self, mask: ArrayLike) -> None:
        """Leave out the elements where the mask is non-zero.

        Args:
            mask: One entry per element.

        Raises:
            ValueError: The mask does not have one entry per element.
        """
        element: cython.long
        set_index: cython.long
        absent_of: cython.long[::1] = np.ascontiguousarray(mask, dtype=np.int64)

        if absent_of.shape[0] != self.n:
            msg = "the mask must have one entry per element"
            raise ValueError(msg)

        for set_index in range(self.n_sets):
            self.head[set_index] = -1
            self.size[set_index] = 0

        for element in range(self.n):
            self.absent[element] = absent_of[element]
            self._link(element, self.target[element])

    @cython.cfunc
    @cython.exceptval(check=False)
    def _zero(self, values: cython.long[::1]) -> cython.void:
        """Set every per-constraint entry to 0.

        Args:
            values: A per-constraint table.
        """
        i: cython.long

        for i in range(self.n_names):
            values[i] = 0

    @cython.cfunc
    @cython.exceptval(check=False)
    def _count_full(self) -> cython.void:
        """Count every instance of the ``act`` constraints into ``cnt``, from zero."""
        self._zero(self.cnt)
        self._full()

    @property
    def full_counts(self) -> list[int]:
        """The raw count per constraint over the whole placement."""
        i: cython.long

        for i in range(self.n_names):
            self.act[i] = 1

        self._count_full()
        return [self.cnt[i] + self.offset[i] for i in range(self.n_names)]

    @cython.ccall
    def full_count(self, constraint_index: cython.long) -> cython.long:
        """Return the raw count of one constraint over the whole placement.

        Args:
            constraint_index: The constraint, as a name position.

        Returns:
            Its count, the instances that read no decision included.
        """
        self._zero(self.act)
        self.act[constraint_index] = 1
        self._count_full()
        return self.cnt[constraint_index] + self.offset[constraint_index]

    @cython.ccall
    def around_set_count(
        self, constraint_index: cython.long, elements: ArrayLike, others: ArrayLike
    ) -> cython.long:
        """A constraint's raw count over the instances those elements can change, each once.

        Args:
            constraint_index: The constraint, as a name position.
            elements: The elements.
            others: Per element, another set whose instances it can change, or -1.

        Returns:
            The count.

        Raises:
            ValueError: ``others`` does not hold one set per element.
        """
        position: cython.long
        element: cython.long
        n_elements: cython.long = 0
        given: cython.long[::1] = np.ascontiguousarray(elements, dtype=np.int64)
        other_of: cython.long[::1] = np.ascontiguousarray(others, dtype=np.int64)

        if other_of.shape[0] != given.shape[0]:
            msg = "others must hold one set per element"
            raise ValueError(msg)

        if self.setlevel[constraint_index]:
            return self.full_count(constraint_index)

        for position in range(given.shape[0]):
            element = given[position]

            if self.scope[element] == 0:
                self.scope[element] = 1
                self.elements[n_elements] = element
                self.others[n_elements] = other_of[position]
                n_elements += 1

        self._zero(self.cnt)
        self._zero(self.act)
        self.act[constraint_index] = 1
        self.sid += 1
        self._around_set(n_elements)

        for position in range(n_elements):
            self.scope[self.elements[position]] = 0

        return self.cnt[constraint_index]

    @cython.cfunc
    @cython.exceptval(check=False)
    def _weighted(self, hard: cython.bint) -> cython.long:
        """The weighted count of the hard (or the soft) constraints.

        Args:
            hard: Whether the hard constraints are counted, else the soft ones.

        Returns:
            The weighted count, the instances that read no decision included.
        """
        i: cython.long
        total: cython.long = 0

        for i in range(self.n_names):
            self.act[i] = 1 if (self.hard[i] != 0) == hard else 0

        self._count_full()

        for i in range(self.n_names):
            if self.act[i]:
                total += self.w[i] * (self.cnt[i] + self.offset[i])

        return total

    @property
    def full_hard(self) -> cython.long:
        """The weighted hard score of the placement."""
        return self._weighted(True)  # noqa: FBT003  a C flag, not an option

    @property
    def full_soft(self) -> cython.long:
        """The weighted soft score of the placement."""
        return self._weighted(False)  # noqa: FBT003  a C flag, not an option

    @cython.cfunc
    @cython.exceptval(check=False)
    def _count_around(
        self,
        element: cython.long,
        constraint_indices: cython.long[::1],
        low: cython.long,
        high: cython.long,
        blame: cython.long[::1],
    ) -> cython.long:
        """The count over ``constraint_indices[low:high]`` of every violation ``element`` is in.

        Args:
            element: The element.
            constraint_indices: The constraints, as name positions.
            low: The first position read in ``constraint_indices``.
            high: The position after the last one read.
            blame: Per constraint, raised by one when it counts a violation here.

        Returns:
            The count.
        """
        position: cython.long
        constraint_index: cython.long
        total: cython.long = 0

        self._zero(self.cnt)
        self._zero(self.act)

        for position in range(low, high):
            self.act[constraint_indices[position]] = 1

        self._around(element, -1)

        for position in range(low, high):
            constraint_index = constraint_indices[position]

            if self.cnt[constraint_index] != 0:
                total += self.cnt[constraint_index]
                blame[constraint_index] += 1

        return total

    @cython.ccall
    def lone_placements(
        self, element: cython.long, candidates: Candidates, filters: Filters
    ) -> LoneDomain:
        """The starts and the sets an element may take alone, every other element absent.

        Args:
            element: The element, absent before and after.
            candidates: The starts and the sets tried.
            filters: The constraints that filter its starts, and those that filter its sets.

        Returns:
            The candidates with no violation alone, and what removed the others.
        """
        start: cython.long
        position: cython.long
        low: cython.long = candidates.low
        high: cython.long = candidates.high
        targets: cython.long[::1] = candidates.targets
        on_start: cython.long[::1] = filters.on_start
        on_set: cython.long[::1] = filters.on_set
        blame: cython.long[::1] = np.zeros(self.n_names, dtype=np.int64)
        starts: list[int] = []
        sets: list[int] = []

        for start in range(low, high):
            if on_start.shape[0] == 0:
                starts.append(start)
                continue

            self.insert(element, start, 0)

            if self._count_around(element, on_start, 0, on_start.shape[0], blame) == 0:
                starts.append(start)

        for position in range(targets.shape[0]):
            if on_set.shape[0] == 0:
                sets.append(targets[position])
                continue

            self.insert(element, starts[0] if starts else low, targets[position])

            if self._count_around(element, on_set, 0, on_set.shape[0], blame) == 0:
                sets.append(targets[position])

        self.remove(element)
        return LoneDomain(starts, sets, np.asarray(blame))

    @staticmethod
    @cython.cfunc
    @cython.exceptval(check=False)
    def _restore(
        mark: cython.long,
        offsets: cython.long[::1],
        pruned: cython.long[::1],
        alive: cython.long[::1],
    ) -> cython.void:
        """Give back the candidates pruned under ``mark``, from position ``mark`` on.

        Args:
            mark: The depth that pruned them, plus one.
            offsets: Where each position's candidates start, plus the end.
            pruned: Per candidate, the mark that pruned it, 0 if alive.
            alive: Per position, its candidates still alive.
        """
        position: cython.long
        candidate: cython.long

        for position in range(mark, offsets.shape[0] - 1):
            for candidate in range(offsets[position], offsets[position + 1]):
                if pruned[candidate] == mark:
                    pruned[candidate] = 0
                    alive[position] += 1

    @staticmethod
    @cython.cfunc
    @cython.inline
    @cython.exceptval(check=False)
    def _n_candidates(domain_offsets: cython.long[::1], position: cython.long) -> cython.long:
        """How many placements an element of a plan may take: its starts times its sets.

        Args:
            domain_offsets: Where each element's starts, then its sets, begin, plus the end.
            position: The element's position in the plan's order.

        Returns:
            The count.
        """
        return (domain_offsets[2 * position + 1] - domain_offsets[2 * position]) * (
            domain_offsets[2 * position + 2] - domain_offsets[2 * position + 1]
        )

    @staticmethod
    @cython.cfunc
    @cython.inline
    @cython.exceptval(check=False)
    def _draw(seed: cython.long) -> cython.long:
        """The next draw of a fixed linear congruential generator: the same seed, the same runs.

        Args:
            seed: The last draw.

        Returns:
            The next one.
        """
        return (seed * 1103515245 + 12345) % 2147483648

    @cython.cfunc
    def _try(
        self,
        element: cython.long,
        position: cython.long,
        candidate: cython.long,
        domains: cython.long[::1],
        domain_offsets: cython.long[::1],
    ) -> cython.void:
        """Place an element at one of its candidates: a start and a set, taken start first.

        Args:
            element: The element.
            position: Its position in the search order.
            candidate: Its candidate, counted from 0.
            domains: Per position, its starts then its sets, one position after another.
            domain_offsets: Where each position's starts, then sets, begin, plus the end.
        """
        low: cython.long = domain_offsets[2 * position]
        middle: cython.long = domain_offsets[2 * position + 1]
        n_targets: cython.long = domain_offsets[2 * position + 2] - middle
        self.insert(
            element, domains[low + candidate // n_targets], domains[middle + candidate % n_targets]
        )

    @cython.ccall
    def depth_first(  # noqa: C901, PLR0912, PLR0914, PLR0915  one depth-first loop
        self, plan: SearchPlan, stop: "StopCondition"
    ) -> SearchOutcome:
        """Search a placement of the plan's group alone together, every other element absent.

        Args:
            plan: The group, its candidates and what each placement reads.
            stop: When to give up, asked at its checkpoints; an iteration is one evaluation.

        Returns:
            The verdict, the evaluations spent and what rejected candidates on the way.
        """
        # depth first; a placed element prunes later candidates its forward checks reject, marked
        # with its depth + 1 and given back when it moves
        order: cython.long[::1] = plan.order
        domains: cython.long[::1] = plan.domains
        domain_offsets: cython.long[::1] = plan.domain_offsets
        checks: cython.long[::1] = plan.checks
        check_offsets: cython.long[::1] = plan.check_offsets
        whole: cython.long[::1] = plan.whole
        blame: cython.long[::1] = np.zeros(self.n_names, dtype=np.int64)

        size: cython.long = order.shape[0]
        depth: cython.long = 0
        work: cython.long = 0
        found: cython.long = 0
        next_check: cython.long = 0

        candidate: cython.long
        later: cython.long
        position: cython.long
        passed: cython.bint

        offsets: cython.long[::1] = np.zeros(size + 1, dtype=np.int64)

        for position in range(size):
            offsets[position + 1] = offsets[position] + Kernel._n_candidates(
                domain_offsets, position
            )

        next_candidate: cython.long[::1] = np.zeros(max(size, 1), dtype=np.int64)
        pruned: cython.long[::1] = np.zeros(max(offsets[size], 1), dtype=np.int64)
        alive: cython.long[::1] = np.diff(np.asarray(offsets)).astype(np.int64)
        group = np.asarray(order)
        unmoved = np.full(size, -1, dtype=np.int64)

        if size == 0:
            return SearchOutcome(1, 0, np.asarray(blame))

        next_candidate[0] = offsets[0]

        while depth >= 0:
            candidate = next_candidate[depth]

            while candidate < offsets[depth + 1] and pruned[candidate] != 0:
                candidate += 1

            next_candidate[depth] = candidate

            if candidate == offsets[depth + 1]:
                self.remove(order[depth])
                depth -= 1

                if depth >= 0:
                    Kernel._restore(depth + 1, offsets, pruned, alive)
                    next_candidate[depth] += 1

                continue

            if work >= next_check:
                if stop.should_stop(work):
                    found = -1
                    break

                next_check = stop.checkpoint(work)

            work += 1
            self._try(order[depth], depth, candidate - offsets[depth], domains, domain_offsets)
            passed = (
                self._count_around(
                    order[depth], checks, check_offsets[depth], check_offsets[depth + 1], blame
                )
                == 0
            )

            if passed and depth == size - 1:
                for position in range(whole.shape[0]):
                    if self.around_set_count(whole[position], group, unmoved) != 0:
                        blame[whole[position]] += 1
                        passed = False

                if passed:
                    found = 1
                    break
            elif passed:
                for later in range(depth + 1, size):
                    if check_offsets[size + later] == check_offsets[size + later + 1]:
                        continue

                    for position in range(offsets[later], offsets[later + 1]):
                        if pruned[position] != 0:
                            continue

                        work += 1
                        self._try(
                            order[later], later, position - offsets[later], domains, domain_offsets
                        )

                        if self._count_around(
                            order[later],
                            checks,
                            check_offsets[size + later],
                            check_offsets[size + later + 1],
                            blame,
                        ):
                            pruned[position] = depth + 1
                            alive[later] -= 1

                    self.remove(order[later])

                    if alive[later] == 0:
                        passed = False
                        Kernel._restore(depth + 1, offsets, pruned, alive)
                        break

            if passed:
                depth += 1
                next_candidate[depth] = offsets[depth]
            else:
                next_candidate[depth] += 1

        for position in range(size):
            self.remove(order[position])

        return SearchOutcome(found, work, np.asarray(blame))

    @cython.cfunc
    def _least(
        self,
        element: cython.long,
        position: cython.long,
        domains: cython.long[::1],
        domain_offsets: cython.long[::1],
        checks: cython.long[::1],
        seed: cython.long,
    ) -> cython.long:
        """The candidate of an element in the fewest ``checks`` violations, ties drawn by ``seed``.

        Args:
            element: The element.
            position: Its position in the search order.
            domains: Per position, its starts then its sets, one position after another.
            domain_offsets: Where each position's starts, then sets, begin, plus the end.
            checks: The constraints counted, as name positions.
            seed: The draw among ties.

        Returns:
            The candidate, counted from 0; the element is left at the last one tried.
        """
        candidate: cython.long
        count: cython.long
        best: cython.long = 0
        least: cython.long = -1
        ties: cython.long = 0
        blame: cython.long[::1] = self.before

        n_candidates: cython.long = Kernel._n_candidates(domain_offsets, position)

        for candidate in range(n_candidates):
            self._try(element, position, candidate, domains, domain_offsets)
            count = self._count_around(element, checks, 0, checks.shape[0], blame)

            if least < 0 or count < least:
                best, least, ties = candidate, count, 1
            elif count == least:
                ties += 1
                seed = Kernel._draw(seed)

                if seed % ties == 0:
                    best = candidate

        return best

    @cython.ccall
    def min_conflicts(
        self, plan: SearchPlan, stop: "StopCondition", seed: cython.long
    ) -> cython.long:
        """Look for a placement of the plan's group by local search: 1 found, -1 not.

        Args:
            plan: The group, placed greedily in order then moved; it reads the closed constraints.
            stop: When to give up, asked at its checkpoints; an iteration is one evaluation.
            seed: The first random draw: the same seed, the same moves.

        Returns:
            1 when no element is left in a violation, else -1: nothing proven either way.
        """
        # min-conflicts: a violated element moves to its least violated candidate (1 in 10: random)
        order: cython.long[::1] = plan.order
        domains: cython.long[::1] = plan.domains
        domain_offsets: cython.long[::1] = plan.domain_offsets
        checks: cython.long[::1] = plan.closed
        size: cython.long = order.shape[0]
        work: cython.long = 0
        next_check: cython.long = 0
        found: cython.long = -1

        position: cython.long
        candidate: cython.long
        n_candidates: cython.long
        n_violated: cython.long
        violated: cython.long[::1] = np.zeros(max(size, 1), dtype=np.int64)
        blame: cython.long[::1] = self.before

        for position in range(size):
            self._try(
                order[position],
                position,
                self._least(order[position], position, domains, domain_offsets, checks, seed),
                domains,
                domain_offsets,
            )
            work += Kernel._n_candidates(domain_offsets, position)

        while True:
            if work >= next_check:
                if stop.should_stop(work):
                    break

                next_check = stop.checkpoint(work)

            n_violated = 0

            for position in range(size):
                if self._count_around(order[position], checks, 0, checks.shape[0], blame):
                    violated[n_violated] = position
                    n_violated += 1

            work += size

            if n_violated == 0:
                found = 1
                break

            seed = Kernel._draw(seed)
            position = violated[seed % n_violated]
            n_candidates = Kernel._n_candidates(domain_offsets, position)

            seed = Kernel._draw(seed)
            candidate = (seed // 10) % n_candidates

            if seed % 10 != 0:
                work += n_candidates
                candidate = self._least(
                    order[position], position, domains, domain_offsets, checks, seed
                )

            self._try(order[position], position, candidate, domains, domain_offsets)

        for position in range(size):
            self.remove(order[position])

        return found

    @cython.ccall
    def delta_place(
        self, element: cython.long, value: cython.long, target: cython.long
    ) -> list[int]:
        """Return the score change of placing ``element``: ``[hard, soft]``, soft 0 unless soft mode.

        Args:
            element: The element.
            value: Its new value.
            target: Its new set.

        Returns:
            The change, compared hard first; the placement is left as it was.
        """
        self._check(element, target)
        self._delta(element, value, target)
        return [self.change[0], self.change[1]]

    @cython.cfunc
    @cython.exceptval(check=False)
    def _delta(self, j: cython.long, ns: cython.long, nr: cython.long) -> cython.void:
        """Put in ``change`` the score change of moving ``j`` to ``ns`` in ``nr``, left as it was.

        Args:
            j: The element.
            ns: Its new value.
            nr: Its new set.
        """
        i: cython.long
        os_: cython.long = self.start[j]
        or_: cython.long = self.target[j]

        self._zero(self.cnt)

        for i in range(self.n_names):
            self.act[i] = 1 if (self.hard[i] or self.soft_mode) else 0

        self._around(j, nr)

        for i in range(self.n_names):
            self.before[i] = self.cnt[i]
            self.cnt[i] = 0

        self._place(j, ns, nr)
        self._around(j, or_)
        self._place(j, os_, or_)
        self._change_from(self.before)

    @cython.cfunc
    @cython.exceptval(check=False)
    def _change_from(self, before: cython.long[::1]) -> cython.void:
        """Put in ``change`` the weighted counts minus ``before``: hard, then soft in soft mode.

        Args:
            before: Per constraint, the counts taken before the move.
        """
        i: cython.long
        self.change[0] = 0
        self.change[1] = 0

        for i in range(self.n_names):
            if self.hard[i]:
                self.change[0] += self.w[i] * (self.cnt[i] - before[i])
            elif self.soft_mode:
                self.change[1] += self.w[i] * (self.cnt[i] - before[i])

    @cython.ccall
    def element_hard(self, element: cython.long) -> cython.long:
        """The weighted hard count of every violation a present element is in; 0 when absent.

        Args:
            element: The element.

        Returns:
            The count.
        """
        i: cython.long
        total: cython.long = 0

        if self.absent[element] != 0:
            return 0

        self._zero(self.cnt)

        for i in range(self.n_names):
            self.act[i] = 1 if self.hard[i] != 0 else 0

        self._around(element, -1)

        for i in range(self.n_names):
            if self.act[i]:
                total += self.w[i] * self.cnt[i]

        return total

    @property
    def violating(self) -> list[int]:
        """The present elements in a hard violation, ascending."""
        return [element for element in range(self.n) if self.element_hard(element) > 0]

    @cython.cfunc
    @cython.exceptval(check=False)
    def _counts_around(
        self, j: cython.long, r_other: cython.long, into: cython.long[::1]
    ) -> cython.void:
        """Copy into ``into`` the scored counts around ``j``, ``r_other`` held as well.

        Args:
            j: The element, present.
            r_other: The set it is about to join, or the one it left.
            into: Per constraint, where the counts go.
        """
        i: cython.long
        self._zero(self.cnt)
        self._around(j, r_other)

        for i in range(self.n_names):
            into[i] = self.cnt[i]

    @cython.cfunc
    @cython.exceptval(check=False)
    def _change_since(
        self, j: cython.long, r_other: cython.long, before: cython.long[::1]
    ) -> cython.void:
        """Put in ``change`` the score change around ``j`` since ``before`` were taken.

        Args:
            j: The element, present where it was moved.
            r_other: The set it left.
            before: Per constraint, the counts taken before it moved.
        """
        self._zero(self.cnt)
        self._around(j, r_other)
        self._change_from(before)

    @cython.ccall
    def relocate(  # noqa: C901, PLR0912, PLR0914, PLR0915  one scan, kept whole for C speed
        self, element: cython.long, partner: cython.long, domains: Domains, draw: cython.double
    ) -> list[int]:
        """Move an element to one of its best placements, a partner to the same start with it.

        Args:
            element: The element, present or absent; present after, unless it has no placement.
            partner: An element placed at the same start, in its best set, or -1.
            domains: Per element, the starts then the sets it may take.
            draw: Uniform in ``[0, 1)``: which of the equally best placements is taken.

        Returns:
            The score change from where they stood, made present: ``[hard, soft]``.
        """
        values: cython.long[::1] = domains.values
        offsets: cython.long[::1] = domains.offsets
        first: cython.long = offsets[2 * element]
        middle: cython.long = offsets[2 * element + 1]
        last: cython.long = offsets[2 * element + 2]
        partner_first: cython.long = 0
        partner_last: cython.long = 1
        was_absent: cython.long = self.absent[element]
        partner_absent: cython.long = 0
        old_start: cython.long = self.start[element]
        old_target: cython.long = self.target[element]
        partner_start: cython.long = 0
        partner_target: cython.long = 0
        best_hard: cython.long = 0
        best_soft: cython.long = 0
        n_kept: cython.long = 0

        hard: cython.long
        soft: cython.long
        lead_hard: cython.long
        lead_soft: cython.long
        i: cython.long
        p: cython.long
        q: cython.long
        pp: cython.long
        s: cython.long
        r: cython.long
        r2: cython.long
        k: cython.long

        if partner >= 0:
            partner_first = offsets[2 * partner + 1]
            partner_last = offsets[2 * partner + 2]
            partner_absent = self.absent[partner]
            partner_start = self.start[partner]
            partner_target = self.target[partner]

        capacity: cython.long = max(
            (middle - first) * (last - middle) * max(partner_last - partner_first, 1), 1
        )

        kept_start: cython.long[::1] = np.empty(capacity, dtype=np.int64)
        kept_target: cython.long[::1] = np.empty(capacity, dtype=np.int64)
        kept_partner: cython.long[::1] = np.empty(capacity, dtype=np.int64)
        before: cython.long[::1] = np.zeros(self.n_names, dtype=np.int64)
        partner_before: cython.long[::1] = np.zeros(self.n_names, dtype=np.int64)

        # deltas are read between present placements: an absent element first stands where it was
        if was_absent:
            self.insert(element, old_start, old_target)

        if partner_absent:
            self.insert(partner, partner_start, partner_target)

        for i in range(self.n_names):
            self.act[i] = 1 if (self.hard[i] or self.soft_mode) else 0

        for q in range(middle, last):
            r = values[q]
            self._counts_around(element, r, before)

            for p in range(first, middle):
                s = values[p]
                self._place(element, s, r)
                self._change_since(element, old_target, before)
                lead_hard, lead_soft = self.change[0], self.change[1]

                for pp in range(partner_first, partner_last):
                    r2 = values[pp] if partner >= 0 else -1
                    hard, soft = lead_hard, lead_soft

                    if partner >= 0:
                        self._counts_around(partner, r2, partner_before)
                        self._place(partner, s, r2)
                        self._change_since(partner, partner_target, partner_before)
                        hard += self.change[0]
                        soft += self.change[1]
                        self._place(partner, partner_start, partner_target)

                    if n_kept == 0 or hard < best_hard or (hard == best_hard and soft < best_soft):
                        best_hard, best_soft = hard, soft
                        n_kept = 0

                    if hard == best_hard and soft == best_soft:
                        kept_start[n_kept] = s
                        kept_target[n_kept] = r
                        kept_partner[n_kept] = r2
                        n_kept += 1

            self._place(element, old_start, old_target)

        if n_kept == 0:
            if was_absent:
                self.remove(element)

            if partner_absent:
                self.remove(partner)

            return [0, 0]

        k = min(max(cython.cast(cython.long, draw * n_kept), 0), n_kept - 1)
        self._place(element, kept_start[k], kept_target[k])

        if partner >= 0:
            self._place(partner, kept_start[k], kept_partner[k])

        return [best_hard, best_soft]
