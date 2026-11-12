"""Binds a model to a data set: each statement once, its data gathered per instance in columns."""

import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from baseline_core.core.optimization.problem_formulation.declaration import (
    Aggregate,
    AssignedGroup,
    Collection,
    FilteredSet,
    IndexSet,
    Loop,
    Member,
    Parameter,
    ParameterRead,
    Position,
    Size,
    SortedList,
    Subset,
    SubsetFamily,
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
from baseline_core.core.optimization.problem_formulation.model import Model
from baseline_core.core.optimization.problem_formulation.term import Term
from baseline_core.core.optimization.solver.binding.bound_collection import (
    BoundFilteredSet,
    ComputedSubset,
    FixedSubset,
    PartitionPart,
)
from baseline_core.core.optimization.solver.binding.schema_reader import SchemaReader
from baseline_core.core.optimization.solver.kernel.shape import BoundStatement, TermShape


@dataclass(frozen=True, slots=True)
class BoundData:
    """The problem's data as the model reads it: positions, values, subsets, statements bound."""

    statements: list[BoundStatement]
    """Per statement, its probe and the data of each of its instances."""
    elements: list[Any]
    """The schema objects of the decided elements, in position order."""
    targets: list[Any]
    """The schema objects of the assignment's targets, in position order; none without one."""
    values: dict[Parameter[Any], NDArray[Any]]
    """Per parameter read, its values."""
    subsets: dict[SubsetFamily[Any], list[list[int]]]
    """Per family read, its subsets as positions."""
    keys: dict[SubsetFamily[Any], list[Hashable]]
    """Per grouping read, the key of each of its subsets."""


@dataclass(frozen=True, slots=True)
class Expansion:
    """The work of one ``expand``: the data, and what is made once while binding it."""

    reader: SchemaReader
    """The schema object, its reads kept."""
    parts: list[PartitionPart]
    """Per target, the elements the search assigns to it."""
    values: dict[Parameter[Any], NDArray[Any]] = field(default_factory=dict)
    """Per parameter read, its values."""
    subsets: dict[SubsetFamily[Any], list[list[int]]] = field(default_factory=dict)
    """Per family read, its subsets as positions."""
    collections: dict[tuple[int, tuple[int, ...]], Any] = field(default_factory=dict)
    """Per collection and loops, by ``id``: its bound collection."""
    flats: dict[int, list[NDArray[np.int64]]] = field(default_factory=dict)
    """Per family read at a computed index, by ``id``: its members and offsets."""
    fixed: dict[tuple[int, int], FixedSubset] = field(default_factory=dict)
    """Per family and position, by ``id``: the subset, made once."""
    rows: dict[int, NDArray[np.int64]] = field(default_factory=dict)
    """Per family read at a computed index, by ``id``: the elements its rows hold."""
    columns: dict[int, Any] = field(default_factory=dict)
    """Per probe number or collection, by ``id``: one value for all instances, or one each."""


class Binder:
    """Binds a model's statements to a schema object: each statement once, over all instances."""

    PROBE_VALUE: ClassVar[float] = float("nan")
    """What a data value reads as in the probe: unknown, so no generated code can depend on it."""

    def __init__(self, model: Model[Any]) -> None:
        """Bind ``model``, whose decisions name the elements.

        Args:
            model: The model.

        Raises:
            ValueError: The model has no decision.
        """
        elements = model.decided_set

        if elements is None:
            msg = "the model needs an m.var, an m.assign, or both"
            raise ValueError(msg)

        self.model = model
        self.decided = elements

    def expand(self, problem: BaseModel) -> BoundData:
        """Bind every statement of the model to ``problem``.

        Args:
            problem: The schema object.

        Returns:
            The statements bound, and the values read: the resources' and decisions' included.
        """
        reader = SchemaReader(problem)
        assignment = self.model.assignment
        targets = [] if assignment is None else reader.elements(assignment.to)
        work = Expansion(reader, [PartitionPart(index) for index in range(max(len(targets), 1))])

        statements = [self._statement(term, work) for term in self.model.terms]
        self._read_declared(work)

        return BoundData(
            statements=statements,
            elements=reader.elements(self.decided),
            targets=targets,
            values=work.values,
            subsets=work.subsets,
            keys={
                family: reader.keys(family) or []
                for family in work.subsets
                if family.by is not None
            },
        )

    def _read_declared(self, work: Expansion) -> None:
        """Read the data of the decisions and resources, which the statements may leave unread.

        Args:
            work: The expansion.
        """
        interval, assignment = self.model.integer_decision, self.model.assignment
        families = [] if assignment is None or assignment.among is None else [assignment.among]
        parameters = [] if interval is None or interval.span is None else [interval.span]

        for resource in self.model.resources:
            transition = resource.transition
            waits = [] if transition is None else [transition.same, transition.other]
            values = (resource.demand, resource.capacity, resource.mask, *waits)
            parameters.extend(value for value in values if isinstance(value, Parameter))

            if isinstance(resource.groups, SubsetFamily):
                families.append(resource.groups)

        for parameter in parameters:
            self._values(parameter, work)

        for family in families:
            self._subsets(family, work)

    def _statement(self, term: Term, work: Expansion) -> BoundStatement:
        """A statement bound once over all its instances: one per combination of its loops.

        Args:
            term: The statement.
            work: The expansion.

        Returns:
            Its probe (loops and data unknown) and, per instance, its constants and collections.
        """
        loops = self._loops(term)
        sizes = [
            len(work.reader.elements(over))
            if isinstance(over, IndexSet)
            else len(self._subsets(over, work))
            for over in loops.values()
        ]
        count = math.prod(sizes)
        grid = np.indices(sizes, dtype=np.int64).reshape(len(sizes), count)
        env = dict(zip(loops, grid, strict=True))

        scope = [self._collection(scoped, env, work) for scoped in term.scope]
        probe = Term(
            expr=self._expr(term.expr, env, work),
            name=term.name,
            violation=term.violation,
            scope=scope,
            label=term.label,
        )

        shape = TermShape(probe.expr)
        scopes = [
            [
                element
                for bound in row
                if isinstance(bound, FixedSubset)
                for element in bound.elements
            ]
            for row in self._gather(scope, count, object, work).tolist()
        ]

        return BoundStatement(
            probe,
            shape,
            self._gather(shape.constants, count, np.float64, work),
            self._gather(shape.collections, count, object, work).tolist(),
            scopes,
            [work.rows.get(id(read.subset)) for read in shape.computed],
        )

    @staticmethod
    def _gather(nodes: list[Any], count: int, dtype: type, work: Expansion) -> NDArray[Any]:
        """The per-instance values of bound nodes: one row per instance, one column per node.

        Args:
            nodes: Constants or collections of a statement's probe.
            count: How many instances the statement has.
            dtype: The type of the values.
            work: The expansion.

        Returns:
            The values.
        """
        values: NDArray[Any] = np.empty((count, len(nodes)), dtype=dtype)

        for column, node in enumerate(nodes):
            values[:, column] = work.columns[id(node)]

        return values

    def _loops(self, term: Term) -> dict[int, IndexSet[Any] | SubsetFamily[Any]]:
        """The loops a statement runs over, by symbol, in the order they appear.

        Args:
            term: The statement.

        Returns:
            What each loop ranges over.
        """
        found: dict[int, IndexSet[Any] | SubsetFamily[Any]] = {}
        self._find_loops(term.expr, found)

        for scoped in term.scope:
            self._find_loops(scoped, found)

        return found

    def _find_loops(
        self, node: object, found: dict[int, IndexSet[Any] | SubsetFamily[Any]]
    ) -> None:
        """Collect the loops under a node or a collection.

        Args:
            node: An expression or a collection.
            found: Where the loops are added.
        """
        match node:
            case Loop(over=over, symbol=symbol):
                found.setdefault(symbol, over)
            case Member(collection=inner) | Size(collection=inner) | Position(sorted_list=inner):
                self._find_loops(inner, found)
            case Expr():
                for child in node.children:
                    self._find_loops(child, found)
            case Subset(key=inner) | AssignedGroup(target=inner):
                self._find_loops(inner, found)
            case FilteredSet(base=inner) | SortedList(base=inner):
                self._find_loops(inner, found)

    def _subsets(self, family: SubsetFamily[Any], work: Expansion) -> list[list[int]]:
        """The subsets of a family, as positions, read once.

        Args:
            family: The family.
            work: The expansion.

        Returns:
            One list per subset.
        """
        if family not in work.subsets:
            if family.through is not None:
                rows = self._subsets(family.through.base, work)
                work.subsets[family] = [
                    rows[position]
                    for position in self._values(family.through.index, work).tolist()
                ]
            else:
                work.subsets[family] = work.reader.read(family)

        return work.subsets[family]

    def _values(self, param: Parameter[Any], work: Expansion) -> NDArray[Any]:
        """A parameter's values, read once: joined on another, from the model or from the data.

        Args:
            param: The parameter.
            work: The expansion.

        Returns:
            Its values as data the expressions index.
        """
        if param not in work.values:
            if param.through is not None:
                base, index = param.through.base, param.through.index
                values = self._values(base, work)[self._values(index, work)]
            elif param.values is not None:
                values = np.asarray(param.values, dtype=np.int64)
            else:
                values = np.asarray(work.reader.read(param))

            work.values[param] = values

        return work.values[param]

    def _expr(  # noqa: C901, PLR0911, PLR0912, PLR0914  one branch per node class
        self, expr: Expr, env: dict[int, NDArray[np.int64]], work: Expansion
    ) -> Expr:
        """The probe of an expression, each of its numbers and collections noted per instance.

        Args:
            expr: The expression as written.
            env: The values of each loop, one per instance.
            work: The expansion.

        Returns:
            The bound expression, its data unknown: what the generated code may rely on.

        Raises:
            ValueError: A node class the binder does not know.
        """
        match expr:
            case Arithmetic(operator=operator, left=left, right=right):
                left, right = self._expr(left, env, work), self._expr(right, env, work)
                return Arithmetic(operator, left, right)
            case Comparison(operator=operator, left=left, right=right):
                left, right = self._expr(left, env, work), self._expr(right, env, work)
                return Comparison(operator, left, right)
            case Logical(operator=operator, left=left, right=right):
                left, right = self._expr(left, env, work), self._expr(right, env, work)
                return Logical(operator, left, right)
            case Constant(value=value):
                work.columns[id(expr)] = value
                return expr
            case ValueRead(decision=decision, element=element):
                return ValueRead(decision, self._expr(element, env, work))
            case Member(collection=collection, symbol=symbol):
                return Member(self._collection(collection, env, work), symbol)
            case ParameterRead(parameter=parameter, element=element):
                return self._param(parameter, self._expr(element, env, work), work)
            case Loop(symbol=symbol):
                return self._unknown(env[symbol], work)
            case Lookup(data=data, index=index):
                return Lookup(data, self._expr(index, env, work))
            case TargetRead(assignment=assignment, element=element):
                return TargetRead(assignment, self._expr(element, env, work))
            case Not(operand=operand):
                return Not(self._expr(operand, env, work))
            case IfThenElse(condition=condition, when_true=when_true, when_false=when_false):
                return IfThenElse(
                    self._expr(condition, env, work),
                    self._expr(when_true, env, work),
                    self._expr(when_false, env, work),
                )
            case Aggregate(operator=operator, member=member, body=body):
                bound = Member(self._collection(member.collection, env, work), member.symbol)
                return Aggregate(operator, bound, self._expr(body, env, work))
            case Size(collection=collection):
                return Size(self._collection(collection, env, work))
            case Position(sorted_list=sorted_list, offset=offset):
                return Position(self._collection(sorted_list, env, work), offset)

        msg = f"node {type(expr).__name__}"
        raise ValueError(msg)

    def _unknown(self, values: float | NDArray[Any], work: Expansion) -> Constant:
        """A number of the probe, unknown, whose value per instance is noted.

        Args:
            values: Its value in every instance, or one per instance.
            work: The expansion.

        Returns:
            The probe's number.
        """
        constant = Constant(self.PROBE_VALUE)
        work.columns[id(constant)] = values
        return constant

    def _param(self, param: Parameter[Any], element: Expr, work: Expansion) -> Expr:
        """A parameter read at a bound element: a lookup, or a number when the element is fixed.

        Args:
            param: The parameter.
            element: The element, already bound.
            work: The expansion.

        Returns:
            The node.
        """
        if not isinstance(element, Constant):
            return Lookup(self._values(param, work), element)

        positions = np.asarray(work.columns[id(element)]).astype(np.int64)
        return self._unknown(self._values(param, work)[positions].astype(np.float64), work)

    def _collection(
        self, collection: object, env: dict[int, NDArray[np.int64]], work: Expansion
    ) -> Any:  # noqa: ANN401  any bound collection
        """The probe of a collection, made once per statement loops, its value per instance noted.

        Args:
            collection: The collection as written.
            env: The values of each loop, one per instance.
            work: The expansion.

        Returns:
            A fixed subset, a decided set, a filtered or sorted set, or a subset at an index.

        Raises:
            ValueError: The collection cannot be made concrete.
        """
        key = (id(collection), tuple(env))

        if key in work.collections:
            return work.collections[key]

        out: Collection[Any] | SortedList
        values: object = None

        match collection:
            case IndexSet():
                count = len(work.reader.elements(collection))
                out = FixedSubset(list(range(count)), decision=collection is self.decided)
            case Subset(family=family, key=selector):
                index = self._expr(selector, env, work)

                if isinstance(index, Constant):
                    out = FixedSubset([], decision=family.of is self.decided)
                    values = self._fixed(family, index, work)
                else:
                    flat, offsets = self._flat(family, work)
                    out = ComputedSubset(flat, offsets, index)

                    if family.of is self.decided:
                        work.rows[id(out)] = np.unique(flat[: offsets[-1]])
            case AssignedGroup(target=target):
                out = PartitionPart(0)
                values = [work.parts[position] for position in env[target.symbol].tolist()]
            case FilteredSet(base=base, mask=mask):
                out = BoundFilteredSet(self._collection(base, env, work), self._values(mask, work))
            case SortedList(base=base, key=decision):
                out = SortedList(self._collection(base, env, work), decision)
            case FixedSubset() | PartitionPart() | ComputedSubset():
                out = collection
            case _:
                msg = f"collection {type(collection).__name__}"
                raise ValueError(msg)

        work.collections[key] = out
        work.columns[id(out)] = out if values is None else values
        return out

    def _fixed(
        self, family: SubsetFamily[Any], index: Constant, work: Expansion
    ) -> FixedSubset | list[FixedSubset]:
        """The subset of a family an index names, per instance; each subset is made once.

        Args:
            family: The family.
            index: The probe's number naming the subset.
            work: The expansion.

        Returns:
            The subset of every instance, or one per instance.
        """
        positions = np.asarray(work.columns[id(index)]).astype(np.int64)
        fixed = []

        for position in positions.ravel().tolist():
            if (id(family), position) not in work.fixed:
                work.fixed[id(family), position] = FixedSubset(
                    list(self._subsets(family, work)[position]), decision=family.of is self.decided
                )

            fixed.append(work.fixed[id(family), position])

        return fixed if positions.ndim else fixed[0]

    def _flat(self, family: SubsetFamily[Any], work: Expansion) -> list[NDArray[np.int64]]:
        """A family's subsets one after another, and where each starts, made once.

        Args:
            family: The family.
            work: The expansion.

        Returns:
            ``[members, offsets]``, the offsets one more than the subsets.
        """
        if id(family) not in work.flats:
            groups = self._subsets(family, work)
            sizes = [len(group) for group in groups]
            offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            flat = np.array(
                [member for group in groups for member in group] or [0], dtype=np.int64
            )
            work.flats[id(family)] = [flat, offsets]

        return work.flats[id(family)]
