"""A bound term's shape and the instances sharing it: one block of generated code for them all."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from baseline_core.core.optimization.problem_formulation.declaration import (
    Aggregate,
    Collection,
    Member,
    Position,
    Size,
    SortedList,
    TargetRead,
    ValueRead,
)
from baseline_core.core.optimization.problem_formulation.expression import (
    Arithmetic,
    Comparison,
    Constant,
    Expr,
    IfThenElse,
    Logical,
    Lookup,
    Not,
)
from baseline_core.core.optimization.problem_formulation.term import Term
from baseline_core.core.optimization.solver.binding.bound_collection import (
    BoundFilteredSet,
    ComputedSubset,
    FixedSubset,
    PartitionPart,
)


class TermKind(StrEnum):
    """What one evaluation of a term ranges over.

    Attributes:
        MEMBER: One free member: evaluated once per member of its collection.
        PAIR: Two free members of one collection: once per pair.
        GROUP: No free member: once per instance, over whole collections.
        SEQUENCE: Consecutive members of a sorted list: once per consecutive pair.
    """

    MEMBER = "member"
    PAIR = "pair"
    GROUP = "group"
    SEQUENCE = "sequence"


@dataclass(frozen=True, slots=True)
class Ranging:
    """What a term's evaluations range over."""

    kind: TermKind
    """Its kind."""
    domain: Collection[Any] | None
    """The collection its evaluations range over; ``None`` for a group."""


@dataclass(frozen=True, slots=True)
class ComputedRead:
    """A family read at a computed index, as a term reads it."""

    subset: ComputedSubset
    """The family, read at the index."""
    mask: NDArray[Any] | None
    """The mask of the filtered set it is the base of, if any."""


@dataclass(frozen=True, slots=True)
class Cells:
    """Per instance and column, which fixed subset or decided set it reads."""

    table: NDArray[np.int64]
    """Per instance and column, a subset's rank in ``subsets``, or ``-1 - set``."""
    subsets: list[FixedSubset]
    """The fixed subsets, each once, in rank order."""


class TermShape:
    """One walk of a bound term: equal keys mean the terms share generated code."""

    PAIR: ClassVar[int] = 2
    """Free members of a pair term."""

    def __init__(self, expr: Expr) -> None:
        """Walk the expression and classify what it ranges over.

        Args:
            expr: The term's expression, bound to data.
        """
        self.constants: list[Constant] = []
        """The numbers, in walk order: each a column of per-instance data the shared code reads."""
        self.collections: list[FixedSubset | PartitionPart] = []
        """The domain first (if any), then the aggregates' collections, in walk order."""
        self.masks: list[NDArray[Any] | None] = []
        """The mask of each of ``collections`` (``None`` where it is not filtered)."""
        self.computed: list[ComputedRead] = []
        """The families read at a computed index, in walk order, each with its mask."""
        self.symbol_order: dict[int, int] = {}
        self.aggregated_symbols: set[int] = set()
        self.free: list[Member[Any]] = []
        self.sorted_list: SortedList | None = None
        self.element_ref_slots: list[int] = []
        self.reads_decision = False
        """Whether it reads a decision, so its count changes with the placement."""

        body = self._walk(expr)
        ranging = self._classify(self.sorted_list, self.free)
        self.kind, self.domain = ranging.kind, ranging.domain
        head = self._collection_key(self.domain, front=True) if self.domain is not None else ""
        self.reads_decision = self.reads_decision or self.on_sets
        self.key = f"{self.kind}:{head}:{body}"

    @staticmethod
    def _classify(sorted_list: SortedList | None, free: list[Member[Any]]) -> Ranging:
        """What a term ranges over, from the sorted list and the free members its walk found.

        Args:
            sorted_list: The sorted list it reads, if any.
            free: Its free members.

        Returns:
            Its kind, and the collection its evaluations range over.

        Raises:
            ValueError: More than two free members, or a pair from two collections.
        """
        if sorted_list is not None:
            return Ranging(TermKind.SEQUENCE, sorted_list.base)

        if len(free) == 1:
            return Ranging(TermKind.MEMBER, free[0].collection)

        if len(free) == TermShape.PAIR:
            if free[0].collection is not free[1].collection:
                msg = "both members of a pair must come from the same collection"
                raise ValueError(msg)

            return Ranging(TermKind.PAIR, free[0].collection)

        if not free:
            return Ranging(TermKind.GROUP, None)

        msg = "at most two free members per expression"
        raise ValueError(msg)

    @property
    def base(self) -> FixedSubset | PartitionPart | ComputedSubset | None:
        """The unfiltered collection the term ranges over (its first aggregate's for a group)."""
        domain = self.domain if self.domain is not None else next(iter(self.collections), None)
        return (
            domain.base  # type: ignore[return-value]
            if isinstance(domain, BoundFilteredSet)
            else domain
        )

    @property
    def mask(self) -> NDArray[Any] | None:
        """The mask of the domain when it is filtered."""
        return self.domain.mask if isinstance(self.domain, BoundFilteredSet) else None

    @property
    def on_sets(self) -> bool:
        """Whether the term ranges over a decided set (its instances follow the assignment)."""
        return isinstance(self.base, PartitionPart)

    @property
    def only_masked(self) -> bool:
        """Whether all collections it reads share one mask (and it reads no fixed element)."""
        kinds = {id(mask) if mask is not None else None for mask in self.masks}
        alone = not self.computed and not self.element_ref_slots
        return len(kinds) == 1 and None not in kinds and alone

    @staticmethod
    def _place[Item](items: list[Item], item: Item, *, front: bool) -> None:
        """Put an item first (the domain) or last.

        Args:
            items: The list.
            item: The item.
            front: Whether it goes first.
        """
        if front:
            items.insert(0, item)
        else:
            items.append(item)

    def _collection_key(
        self, collection: object, *, front: bool = False, mask: NDArray[Any] | None = None
    ) -> str:
        """A collection's key, registering its data and placing it among the term's collections.

        Args:
            collection: A fixed subset, a decided set, a filtered set or a family read at an index.
            front: Whether it is the domain (placed first).
            mask: The mask of the filtered set it is the base of (already placed), if any.

        Returns:
            Its part of the shape key.

        Raises:
            ValueError: The collection has no evaluation scheme.
        """
        if isinstance(collection, BoundFilteredSet):
            bound_mask = collection.mask
            inner = self._collection_key(collection.base, front=front, mask=bound_mask)
            return f"F{id(bound_mask)}<{inner}>"

        if isinstance(collection, ComputedSubset):
            self.computed.append(ComputedRead(collection, mask))
            key = self._walk(collection.key)
            return f"X{id(collection.flat)}.{id(collection.offsets)}[{key}]"

        if isinstance(collection, FixedSubset | PartitionPart):
            self._place(self.collections, collection, front=front)
            self._place(self.masks, mask, front=front)
            return "K" if isinstance(collection, FixedSubset) else "R"

        msg = f"collection {type(collection).__name__}"
        raise ValueError(msg)

    def _walk(self, expr: Expr) -> str:  # noqa: C901, PLR0911, PLR0912, PLR0914  per node class
        """The key of a node, collecting its constants, collections and free members on the way.

        Args:
            expr: The node.

        Returns:
            Its part of the shape key.

        Raises:
            ValueError: A node that only exists before binding.
        """
        match expr:
            case (
                Arithmetic(operator=operator, left=left, right=right)
                | Comparison(operator=operator, left=left, right=right)
                | Logical(operator=operator, left=left, right=right)
            ):
                return f"{operator}({self._walk(left)},{self._walk(right)})"
            case Constant():
                self.constants.append(expr)
                return "C"
            case ValueRead(element=element) | TargetRead(element=element):
                if not isinstance(element, Member | Position | Constant):
                    # the kernel does not know which instances such an element's move changes
                    msg = "a decision read at an element named by data is not supported"
                    raise ValueError(msg)  # noqa: TRY004  a model refused, not a wrong type

                self.reads_decision = True

                if isinstance(element, Constant):
                    self.element_ref_slots.append(len(self.constants))

                tag = "V" if isinstance(expr, ValueRead) else "T"
                return f"{tag}[{self._walk(element)}]"
            case Member(symbol=symbol):
                if symbol not in self.symbol_order:
                    self.symbol_order[symbol] = len(self.symbol_order)

                    if symbol not in self.aggregated_symbols:
                        self.free.append(expr)

                return f"S{self.symbol_order[symbol]}"
            case Lookup(data=data, index=index):
                return f"D{id(data)}[{self._walk(index)}]"
            case Not() | IfThenElse():
                name = type(expr).__name__
                return f"{name}({','.join(self._walk(child) for child in expr.children)})"
            case Aggregate(operator=operator, member=member, body=body):
                self.aggregated_symbols.add(member.symbol)

                if isinstance(member.collection, PartitionPart):
                    self.reads_decision = True

                collection_key = self._collection_key(member.collection)
                return f"agg_{operator}({self._walk(member)};{collection_key};{self._walk(body)})"
            case Size(collection=collection):
                self.reads_decision = True
                return f"size({self._collection_key(collection)})"
            case Position(sorted_list=sorted_list, offset=offset):
                if self.sorted_list is not None and self.sorted_list is not sorted_list:
                    msg = "one sorted list per expression"
                    raise ValueError(msg)

                self.sorted_list, self.reads_decision = sorted_list, True
                return f"Q{offset}"

        msg = f"{type(expr).__name__} is replaced when the model is bound"
        raise ValueError(msg)


@dataclass(slots=True)
class BoundStatement:
    """One statement bound to data: its probe, the probe's shape, and the data of each instance."""

    probe: Term
    """The statement with its loops and its data unknown."""
    shape: TermShape
    """The probe's shape, shared by every instance."""
    constants: NDArray[np.float64]
    """Instances by the shape's numbers."""
    collections: list[list[FixedSubset | PartitionPart]]
    """Per instance, the shape's collections."""
    scopes: list[list[int]]
    """Per instance, the other elements whose partial counts include it."""
    rows: list[NDArray[np.int64] | None]
    """Per family of ``shape.computed``, the elements its rows hold (``None``: not elements)."""


class Template:
    """The instances of one shape under one constraint: one generated block, data and labels."""

    def __init__(self, index: int, name_index: int, statements: list[BoundStatement]) -> None:
        """Gather every instance of one shape.

        Args:
            index: Its rank among the model's templates, which names its arrays.
            name_index: The position of the constraint it is counted under.
            statements: The statements of that shape, the first one's probe standing for all.

        Raises:
            ValueError: A ``scope`` on a term with free members.
        """
        probe = statements[0].probe
        self.index = index
        self.shape = statements[0].shape
        """The probe's shape: what the generated code may rely on, the data left out."""
        self.expr = probe.expr
        self.violation = probe.violation
        self.name = probe.name
        self.name_index = name_index

        self.constants = np.concatenate([statement.constants for statement in statements])
        """Instances by constants, one row per instance."""
        self.collections = [row for statement in statements for row in statement.collections]
        self.scopes = [scope for statement in statements for scope in statement.scopes]

        if self.shape.kind != TermKind.GROUP and any(self.scopes):
            msg = "scope= is only meaningful on terms without free members"
            raise ValueError(msg)

        self.labels = [
            statement.probe.label for statement in statements for _ in statement.constants
        ]

        self.scoped = bool(probe.scope)
        """Whether its statements name a ``scope`` (in the model, whatever the data)."""
        self.rows = statements[0].rows
        """Per family of ``shape.computed``, the elements its rows hold (``None``: none)."""

        # per instance i, entries[offsets[i]:offsets[i + 1]], sorted: e >= 0 is element e,
        # e < 0 every element assigned to set -1 - e
        cells = self._cells()
        self.cells, self.subsets = cells.table, cells.subsets
        """Per instance and column, its fixed subset's rank in ``subsets``, or ``-1 - set``."""
        members, reads = self._entries()
        self.members, self.member_offsets = self._per_instance(*members)
        """Per instance, the domain its free members range over (member and pair terms only)."""
        self.reads, self.read_offsets = self._per_instance(*reads)
        """Per instance, what every evaluation reads besides its free members, masks applied."""

    @property
    def has_free_members(self) -> bool:
        """Whether its evaluations range over free members (a member or a pair term)."""
        return self.shape.kind in {TermKind.MEMBER, TermKind.PAIR}

    @property
    def _domain_column(self) -> int | None:
        """The column of ``collections`` its free members range over, if it is one."""
        free_on_column = isinstance(self.shape.base, FixedSubset | PartitionPart)
        return 0 if self.has_free_members and free_on_column else None

    @property
    def _read_columns(self) -> list[int]:
        """The columns of ``collections`` that hold elements and are read besides free members."""
        return [
            column
            for column, collection in enumerate(self.shape.collections)
            if column != self._domain_column
            and (isinstance(collection, PartitionPart) or collection.decision)
        ]

    @property
    def reads_elements(self) -> bool:
        """Whether an instance may read fixed elements besides its free members (in the model)."""
        fixed = [self.shape.collections[column] for column in self._read_columns]
        rows = [
            read.subset is not self.shape.base or not self.has_free_members
            for read, held in zip(self.shape.computed, self.rows, strict=True)
            if held is not None
        ]

        return (
            any(isinstance(collection, FixedSubset) for collection in fixed)
            or any(rows)
            or bool(self.shape.element_ref_slots)
            or self.scoped
        )

    @property
    def reads_sets(self) -> int:
        """How many decided sets an instance reads besides its free members (in the model)."""
        return sum(
            isinstance(self.shape.collections[column], PartitionPart)
            for column in self._read_columns
        )

    @property
    def zero_columns(self) -> set[int]:
        """The constants that are a literal 0 in the model, the same in every instance."""
        return {
            column for column, constant in enumerate(self.shape.constants) if constant.value == 0
        }

    @property
    def n_instances(self) -> int:
        """How many instances share the shape."""
        return len(self.constants)

    @property
    def set_level(self) -> bool:
        """Whether instances are whole decided sets (their partial count is the full count)."""
        return self.shape.on_sets and self.shape.kind in {TermKind.GROUP, TermKind.SEQUENCE}

    def _entries(self) -> list[list[NDArray[np.int64]]]:
        """Each instance's domain and reads, as ``[entries, instances]`` pairs that may repeat.

        Returns:
            The domain pair, then the read pair.
        """
        owner, column, elements = self._held()
        cells = self.cells
        set_owner, set_column = np.nonzero(cells < 0)
        domain = -1 if self._domain_column is None else self._domain_column

        parts = [
            [elements, owner, column == domain],
            [cells[set_owner, set_column], set_owner, set_column == domain],
            *self._rows_read(),
            self._named(),
        ]
        flat, owners, chosen = (np.concatenate(part) for part in zip(*parts, strict=True))

        return [[flat[chosen], owners[chosen]], [flat[~chosen], owners[~chosen]]]

    def _rows_read(self) -> list[list[NDArray[Any]]]:
        """The elements of the rows of families read at a computed index, for every instance.

        Returns:
            Per family over the elements: its entries, their instances, whether in the domain.
        """
        count, out = self.n_instances, list[list[NDArray[Any]]]()

        for read, held in zip(self.shape.computed, self.rows, strict=True):
            if held is not None:
                mask = read.mask
                kept = held[np.asarray(mask)[held] != 0] if mask is not None else held
                in_domain = self.has_free_members and read.subset is self.shape.base
                out.append([
                    np.tile(kept, count),
                    np.repeat(np.arange(count), len(kept)),
                    np.full(len(kept) * count, in_domain),
                ])

        return out

    def _named(self) -> list[NDArray[Any]]:
        """The elements each instance names: its element references, then its scope.

        Returns:
            The entries, their instances, and whether in the domain (never).
        """
        slots, count = self.shape.element_ref_slots, self.n_instances
        scoped = np.array(
            [
                [element, instance]
                for instance, scope in enumerate(self.scopes)
                for element in scope
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        entries = np.concatenate([self.constants[:, slots].astype(np.int64).ravel(), scoped[:, 0]])
        instances = np.concatenate([np.repeat(np.arange(count), len(slots)), scoped[:, 1]])
        return [entries, instances, np.zeros(len(entries), dtype=bool)]

    def _cells(self) -> Cells:
        """Number the instances' fixed subsets, each once, and note every decided set.

        Returns:
            Per instance and column, a subset's rank or ``-1 - set``; the subsets in rank order.
        """
        cells = np.zeros((self.n_instances, len(self.shape.collections)), dtype=np.int64)
        ranks: dict[int, int] = {}
        subsets: list[FixedSubset] = []

        for instance, row in enumerate(self.collections):
            for place, collection in enumerate(row):
                if isinstance(collection, PartitionPart):
                    cells[instance, place] = -1 - collection.index
                    continue

                if id(collection) not in ranks:
                    ranks[id(collection)] = len(subsets)
                    subsets.append(collection)

                cells[instance, place] = ranks[id(collection)]

        return Cells(cells, subsets)

    def _held(self) -> list[NDArray[np.int64]]:
        """Each element its fixed element subsets hold, masks applied.

        Returns:
            The instance, the column and the element of each.
        """
        lengths = np.array([len(subset.elements) for subset in self.subsets], dtype=np.int64)
        decision = np.array([subset.decision for subset in self.subsets], dtype=bool)
        fixed = self.cells >= 0
        fixed[fixed] = decision[self.cells[fixed]]
        sizes = lengths[self.cells[fixed]]

        starts = (np.cumsum(lengths) - lengths)[self.cells[fixed]]
        flat = [element for subset in self.subsets for element in subset.elements]
        elements = np.array(flat, dtype=np.int64)[
            np.repeat(starts - np.cumsum(sizes) + sizes, sizes) + np.arange(sizes.sum())
        ]

        owner, column = (np.repeat(axis, sizes) for axis in np.nonzero(fixed))

        kept = np.ones(len(elements), dtype=bool)

        for masked, mask in enumerate(self.shape.masks):
            if mask is not None:
                at = column == masked
                kept[at] = np.asarray(mask)[elements[at]] != 0

        return [owner[kept], column[kept], elements[kept]]

    def _per_instance(
        self, entries: NDArray[np.int64], instances: NDArray[np.int64]
    ) -> list[NDArray[np.int64]]:
        """Each instance's entries, sorted and once each, one instance after another.

        Args:
            entries: The entries.
            instances: The instance of each.

        Returns:
            ``[entries, offsets]``, the offsets one more than the instances.
        """
        order = np.lexsort((entries, instances))
        entries, instances = entries[order], instances[order]
        kept = np.ones(len(entries), dtype=bool)
        kept[1:] = (entries[1:] != entries[:-1]) | (instances[1:] != instances[:-1])
        per_instance = np.bincount(instances[kept], minlength=self.n_instances)
        return [entries[kept], np.concatenate([[0], np.cumsum(per_instance)]).astype(np.int64)]
