"""The model a problem is written in: declarations, decisions, constraints and objective terms."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel

from baseline_core.core.optimization.problem_formulation.declaration import (
    Aggregate,
    AssignedGroup,
    Assignment,
    Collection,
    IndexSet,
    IntegerDecision,
    ListMember,
    Member,
    Parameter,
    RecordedPath,
    SchemaPath,
    SortedList,
    SubsetFamily,
    Through,
)
from baseline_core.core.optimization.problem_formulation.expression import (
    Arithmetic,
    Constant,
    Expr,
    IfThenElse,
    Operand,
)
from baseline_core.core.optimization.problem_formulation.term import Term


class Tie(StrEnum):
    """How a search moves the elements one constraint ties.

    Attributes:
        RELEASED: A destroy releases them together.
        PAIRED: Each pair it ties takes one value together.
        GROUPED: They are released together whatever the destroy.
    """

    RELEASED = "released"
    PAIRED = "paired"
    GROUPED = "grouped"


@dataclass(slots=True)
class Transition:
    """The wait between two members of a group one after the other (CP Optimizer transitions)."""

    kind: Parameter[Any]
    """Per element, its kind."""
    same: int | Parameter[Any]
    """What the later member waits after one of its own kind: one number, or per element."""
    other: int | Parameter[Any]
    """What the later member waits after one of another kind: one number, or per element."""


@dataclass(slots=True)
class Resource:
    """A capacity groups of elements share: ``no_overlap``, ``cumulative``, ``bin_packing``."""

    name: str
    """The constraint it is counted under."""
    groups: SubsetFamily[Any] | Assignment[Any] | Collection[Any]
    """A family (a group per subset), an assignment (one group per target), or one collection."""
    interval: IntegerDecision | None
    """The interval an element holds the capacity for; ``None``: for good (``bin_packing``)."""
    demand: int | Parameter[Any]
    """What an element uses: one number, or per element."""
    capacity: int | Parameter[Any]
    """What a group offers at once: one number, or per assignment target."""
    mask: Parameter[Any] | None = None
    """Per element, whether it takes part (``None``: every element)."""
    transition: Transition | None = None
    """The wait between two members one after the other, if any."""


@dataclass(slots=True)
class BreakRule:
    """A break each period a group works, as ``m.breaks`` posts it (OR-Tools vehicle breaks)."""

    name: str
    """The constraint the missing breaks are counted under."""
    groups: SubsetFamily[Any] | Assignment[Any] | Collection[Any]
    """The groups that keep a break: a family, an assignment, or one collection."""
    interval: IntegerDecision
    """The interval decision whose members occupy the slots."""
    mask: Parameter[Any] | None
    """Per element, whether it takes part (``None``: every element)."""
    period: int
    """Slots in a period: periods are ``[k * period, (k + 1) * period)``."""
    window: list[int]
    """``[start, end)``, slots from the start of the period, where the break must lie."""
    length: int
    """Consecutive free slots the break needs."""

    def __post_init__(self) -> None:
        """Refuse a break that no period can hold, or a horizon that is no whole periods.

        Raises:
            ValueError: The period, window, break length or ``high`` hold no break.
        """
        if len(self.window) != 2:  # noqa: PLR2004  a start and an end
            msg = f"a break window is [start, end]; got {self.window}"
            raise ValueError(msg)

        first, last = self.window

        if self.period < 1 or not 0 <= first < last <= self.period:
            msg = f"the break window {self.window} must lie inside a period of {self.period} slots"
            raise ValueError(msg)

        if not 1 <= self.length <= last - first:
            msg = f"a break of {self.length} slots must fit its window {self.window}"
            raise ValueError(msg)

        high = self.interval.high

        if high % self.period:
            msg = (
                f"breaks need m.var(..., high=...) in whole periods of {self.period} slots;"
                f" high is {high}"
            )
            raise ValueError(msg)


class Model[S: BaseModel]:
    """A problem in ``m.`` declarations and expressions: hard constraints, soft terms."""

    def __init__(self, schema: type[S]) -> None:
        """Start an empty model over a schema type.

        Args:
            schema: The type of the data object it will be bound to.
        """
        self._schema_type = schema
        self.integer_decision: IntegerDecision | None = None
        self.assignment: Assignment[Any] | None = None
        self.terms: list[Term] = []
        self.weights: dict[str, int | None] = {}
        self.resources: list[Resource] = []
        self.break_rules: list[BreakRule] = []
        self.ties: dict[Tie, str] = {}

    @property
    def names(self) -> list[str]:
        """The constraint names, in the order they were first posted."""
        return list(self.weights)

    @property
    def decided_set(self) -> IndexSet[Any] | None:
        """The set the decisions are over: the integer decision's, else the assignment's."""
        if self.integer_decision is not None:
            return self.integer_decision.over

        return None if self.assignment is None else self.assignment.of

    @property
    def hard(self) -> list[bool]:
        """Per constraint, in name order, whether ``constraint`` or a primitive posted it."""
        return [weight is None for weight in self.weights.values()]

    @property
    def unit_weights(self) -> list[int]:
        """Per constraint, in name order, its weight in its own axis: 1 for a hard one."""
        return [1 if weight is None else weight for weight in self.weights.values()]

    @property
    def variables(self) -> list[str | None]:
        """``[integer variable, assignment]`` names, ``None`` where the model has none."""
        integer, assignment = self.integer_decision, self.assignment
        return [
            None if integer is None else integer.name,
            None if assignment is None else assignment.name,
        ]

    def _read(self, selector: Callable[[Any], object], over: IndexSet[Any] | None) -> RecordedPath:
        """The path a selector reads, run on the elements of ``over`` (or on the schema root).

        Args:
            selector: The lambda.
            over: The set whose element it receives; ``None``: the schema object.

        Returns:
            The recorded path.
        """
        schema = self._schema_type if over is None else over.schema
        return RecordedPath.of(selector(SchemaPath(RecordedPath(schema, [], over))))

    @staticmethod
    def _own(path: RecordedPath) -> str:
        """A path read on the element itself, no key followed.

        Args:
            path: The recorded path.

        Returns:
            Its name.

        Raises:
            ValueError: The path follows a key.
        """
        if path.key is not None:
            msg = f"{path.name!r} follows a key; here a selector reads the element's own fields"
            raise ValueError(msg)

        return path.name

    def set[T](
        self,
        selector: Callable[[S], Iterable[T]],
        key: str | None = None,
        *,
        distinct: bool = False,
    ) -> IndexSet[T]:
        """Declare the elements a list of the schema holds: ``m.set(lambda s: s.operations)``.

        Args:
            selector: The list, read on the schema object.
            key: The field that names the elements, so ``jobs[op.job]`` finds one.
            distinct: Whether equal elements (copies) count once.

        Returns:
            The set.
        """
        path = self._read(selector, None)
        return IndexSet(self._own(path), path.schema, key, distinct=distinct)

    def param[T](
        self, selector: Callable[[T], object], over: IndexSet[T], of: IndexSet[Any] | None = None
    ) -> Parameter[T]:
        """Declare integer data, one value per element of ``over``.

        Args:
            selector: The value, read on an element; ``jobs[op.job]`` joins on a key.
            over: The set it is indexed by.
            of: The set its values name: a text (or an object) becomes that element's position.

        Returns:
            The parameter.
        """
        return self._parameter(self._read(selector, over), over, of)

    @classmethod
    def _parameter(
        cls, path: RecordedPath, over: IndexSet[Any], of: IndexSet[Any] | None
    ) -> Parameter[Any]:
        """The parameter a path reads per element of ``over``, joined on each key it follows.

        Args:
            path: The recorded path.
            over: The set it is indexed by.
            of: The set its values name, if any.

        Returns:
            The parameter.
        """
        if path.key is None or path.origin is None:
            return Parameter(path.name, over, of)

        index = cls._parameter(path.key, over, path.origin)
        base = Parameter(path.name, path.origin, of)
        return Parameter(f"{base.name}[{index.name}]", over, through=Through(base, index))

    def subsets[T](
        self,
        selector: Callable[[T], Iterable[object]],
        over: IndexSet[T],
        of: IndexSet[Any] | None = None,
    ) -> SubsetFamily[Any]:
        """Declare one subset of ``of`` (or of plain values) per element of ``over``.

        Args:
            selector: The list, read on an element; ``jobs[op.job]`` joins on a key.
            over: The set it is indexed by.
            of: The set its items name: a text (or an object) becomes that element's position.

        Returns:
            The family.
        """
        return self._family(self._read(selector, over), over, of)

    @classmethod
    def _family(
        cls, path: RecordedPath, over: IndexSet[Any], of: IndexSet[Any] | None
    ) -> SubsetFamily[Any]:
        """The family a path reads per element of ``over``, joined on the key it follows.

        Args:
            path: The recorded path.
            over: The set it is indexed by.
            of: The set its items name, if any.

        Returns:
            The family.

        Raises:
            ValueError: The lists are read per element of a ``distinct`` set.
        """
        if path.origin is not None and path.origin.distinct:
            # a family is read from the schema root: one row per copy, not per element kept
            msg = f"{path.name!r} is read per element of a distinct set; drop distinct=True"
            raise ValueError(msg)

        if path.key is None or path.origin is None:
            return SubsetFamily(path.name, of)

        index = cls._parameter(path.key, over, path.origin)
        base = SubsetFamily(path.name, of)
        return SubsetFamily(f"{base.name}[{index.name}]", of, through=Through(base, index))

    def group_by[T](
        self,
        by: Callable[[T], object],
        over: IndexSet[T],
        *,
        same_order_as: SubsetFamily[Any] | None = None,
    ) -> SubsetFamily[T]:
        """One subset of ``over`` per key ``by`` reads; an element with a list of keys joins each.

        Args:
            by: The key, read on an element: one value, or a list of them (empty: no group).
            over: The set whose elements are grouped.
            same_order_as: Another grouping whose order of keys to follow, so both index alike.

        Returns:
            The family of groups.
        """
        path = self._own(self._read(by, over))
        return SubsetFamily("groups:" + path, over, by=path, same_order_as=same_order_as)

    def within[T, U](self, selector: Callable[[T], Iterable[U]], element: Member[T]) -> U:
        """A member of the list read on ``element``: ``window.start`` reads that item's field.

        Args:
            selector: The list, read on an element; ``jobs[op.job]`` joins on a key.
            element: A free member of a set.

        Returns:
            The member, typed as one item of the list.

        Raises:
            ValueError: ``element`` is no set member, or an item field shadows a member attribute.
        """
        over = element.collection

        if not isinstance(over, IndexSet):
            msg = "m.within reads a list on a member of a set (m.set(...).member)"
            raise ValueError(msg)  # noqa: TRY004  a model the engine refuses, not a wrong type

        path = self._read(selector, over)
        clash = set(RecordedPath.fields(path.schema) if path.schema else []) & set(dir(ListMember))

        if clash:
            msg = f"m.within cannot read the fields {sorted(clash)}: the member has them too"
            raise ValueError(msg)

        items: IndexSet[U] = IndexSet(path.name, path.schema)
        family = self._family(path, over, items)
        return cast("U", ListMember(family[element], items=items))

    @staticmethod
    def lookup_table(values: list[int]) -> Parameter[Any]:
        """Integers in the model, read at a position: ``table[kind[element]]``.

        Args:
            values: The value at each position.

        Returns:
            A parameter read at a per-element parameter whose values are positions.

        Raises:
            ValueError: If a value is not a whole number; the kernel counts in integers.
        """
        fractional = [value for value in values if value != int(value)]

        if fractional:
            error_msg = f"lookup table values must be integers, got {fractional}"
            raise ValueError(error_msg)

        return Parameter("table", None, values=[int(value) for value in values])

    def var(
        self,
        name: str,
        over: IndexSet[Any],
        *,
        span: Parameter[Any] | None = None,
        low: int = 0,
        high: int,
        period: int | None = None,
    ) -> IntegerDecision:
        """Declare an integer per element, with ``span`` an interval ``[start, start + span)``.

        Args:
            name: Its name, by which a schedule is loaded.
            over: The elements.
            span: Per element, how long it lasts.
            low: Least value.
            high: Greatest end: the search and the pre-check list every value up to it.
            period: Values per period, if the values come in periods.

        Returns:
            The decision.

        Raises:
            ValueError: The model already has one: the kernel holds one integer per element.
        """
        if self.integer_decision is not None:
            msg = "the model already has an m.var; the kernel holds one m.var per model"
            raise ValueError(msg)

        self.integer_decision = IntegerDecision(name, over, span, low, high, period)
        return self.integer_decision

    def assign[T](
        self, name: str, of: IndexSet[T], to: IndexSet[Any], among: SubsetFamily[Any] | None = None
    ) -> Assignment[T]:
        """Declare that each element of ``of`` goes to exactly one element of ``to``.

        Args:
            name: Its name, by which a schedule is loaded.
            of: The elements assigned.
            to: The targets.
            among: Per element, the targets it may go to.

        Returns:
            The decision.

        Raises:
            ValueError: The model already has one: the kernel holds one target per element.
        """
        if self.assignment is not None:
            msg = "the model already has an m.assign; the kernel holds one m.assign per model"
            raise ValueError(msg)

        assignment = Assignment(name, of, to, among)
        self.assignment = assignment
        return assignment

    @staticmethod
    def largest(*items: Operand) -> Expr:
        """The largest of the arguments.

        Args:
            items: Expressions or numbers.

        Returns:
            The maximum.
        """
        result = Expr.wrap(items[0])

        for item in items[1:]:
            result = Arithmetic("max", result, Expr.wrap(item))

        return result

    @staticmethod
    def smallest(*items: Operand) -> Expr:
        """The smallest of the arguments.

        Args:
            items: Expressions or numbers.

        Returns:
            The minimum.
        """
        result = Expr.wrap(items[0])

        for item in items[1:]:
            result = Arithmetic("min", result, Expr.wrap(item))

        return result

    @staticmethod
    def if_then_else(condition: Expr, when_true: Operand, when_false: Operand) -> Expr:
        """``when_true`` when ``condition`` holds, else ``when_false``.

        Args:
            condition: The condition.
            when_true: The value when it holds.
            when_false: The value otherwise.

        Returns:
            The choice.
        """
        return IfThenElse(condition, Expr.wrap(when_true), Expr.wrap(when_false))

    @staticmethod
    def exists(items: list[Expr]) -> Expr:
        """True when any item is (false for an empty list): MiniZinc ``exists``.

        Args:
            items: The conditions.

        Returns:
            Their disjunction.
        """
        result: Expr = Constant(0.0)

        for position, item in enumerate(items):
            result = item if position == 0 else result | item

        return result

    @staticmethod
    def forall(items: list[Expr]) -> Expr:
        """True when every item is (true for an empty list): MiniZinc ``forall``.

        Args:
            items: The conditions.

        Returns:
            Their conjunction.
        """
        result: Expr = Constant(1.0)

        for position, item in enumerate(items):
            result = item if position == 0 else result & item

        return result

    @staticmethod
    def _free(member: object) -> Member[Any]:
        """The free member an aggregate ranges over.

        Args:
            member: A collection's ``member``, or what ``m.within`` gives.

        Returns:
            The member.

        Raises:
            TypeError: Anything else.
        """
        if not isinstance(member, Member):
            msg = f"an aggregate ranges over a free member, not {type(member).__name__}"
            raise TypeError(msg)

        return member

    @classmethod
    def count(cls, member: object, condition: Expr) -> Expr:
        """How many members of ``member``'s collection satisfy ``condition``.

        Args:
            member: A free member of a collection.
            condition: What it must satisfy.

        Returns:
            The count.
        """
        return Aggregate("count", cls._free(member), condition)

    @classmethod
    def sum(cls, member: object, expr: Operand) -> Expr:
        """``expr`` summed over the members of ``member``'s collection.

        Args:
            member: A free member of a collection.
            expr: The value per member.

        Returns:
            The sum.
        """
        return Aggregate("sum", cls._free(member), Expr.wrap(expr))

    @classmethod
    def min(cls, member: object, expr: Operand) -> Expr:
        """Smallest value of ``expr`` over the members (a very large number when there is none).

        Args:
            member: A free member of a collection.
            expr: The value per member.

        Returns:
            The minimum.
        """
        return Aggregate("min", cls._free(member), Expr.wrap(expr))

    @staticmethod
    def sort(collection: Collection[Any], key: IntegerDecision) -> SortedList:
        """The members of a collection in increasing order of a decision (ties by key order).

        Args:
            collection: The collection.
            key: The decision it is sorted by.

        Returns:
            The sorted list.
        """
        return SortedList(collection, key)

    def no_overlap(
        self,
        groups: SubsetFamily[Any] | Assignment[Any] | Collection[Any],
        interval: IntegerDecision,
        *,
        name: str,
        where: Parameter[Any] | None = None,
        transition: Transition | None = None,
    ) -> None:
        """No two intervals of one group overlap (CP-SAT ``AddNoOverlap``), a hard constraint.

        Args:
            groups: A family (a group per subset), an assignment (one per target), or a collection.
            interval: The interval decision.
            name: The constraint the violations are counted under.
            where: Per element, whether it takes part; ``None``: every member.
            transition: The wait between two members one after the other, if any.
        """
        for group in self._groups(groups):
            member, other = (group.where(where) if where is not None else group).pair
            self.constraint(self._apart(interval, transition, member, other), name=name)

        self.resources.append(Resource(name, groups, interval, 1, 1, where, transition))

    def _apart(
        self,
        interval: IntegerDecision,
        transition: Transition | None,
        member: Member[Any],
        other: Member[Any],
    ) -> Expr:
        """Two members one after the other, the later waiting its transition when there is one.

        Args:
            interval: The interval decision.
            transition: The wait, if any.
            member: A member.
            other: Another member of the same group.

        Returns:
            The condition.
        """
        if transition is None:
            return (interval.end(member) <= interval[other]) | (
                interval.end(other) <= interval[member]
            )

        wait_after = interval[other] - interval.end(member)
        wait_before = interval[member] - interval.end(other)
        same = transition.kind[member] == transition.kind[other]

        other_needs = self.if_then_else(
            same, self._at(transition.same, other), self._at(transition.other, other)
        )
        member_needs = self.if_then_else(
            same, self._at(transition.same, member), self._at(transition.other, member)
        )

        return ((wait_after >= 0) & (wait_after >= other_needs)) | (
            (wait_before >= 0) & (wait_before >= member_needs)
        )

    @staticmethod
    def _at(value: int | Parameter[Any], member: Member[Any]) -> Operand:
        """A number, or a parameter read at a member.

        Args:
            value: The number or the parameter.
            member: The member.

        Returns:
            The number, or the read.
        """
        return value[member] if isinstance(value, Parameter) else value

    def breaks(  # noqa: PLR0913  the primitive and its period, window and length
        self,
        groups: SubsetFamily[Any] | Assignment[Any] | Collection[Any],
        interval: IntegerDecision,
        *,
        name: str,
        period: int,
        window: list[int],
        length: int,
        where: Parameter[Any] | None = None,
    ) -> None:
        """Each period a group works, it keeps ``length`` consecutive free slots in ``window``.

        Args:
            groups: A family (a group per subset), an assignment (one per target), or a collection.
            interval: The interval decision, with ``high`` in whole periods.
            name: The constraint the missing breaks are counted under.
            period: Slots in a period; a group works in a period when a member starts in it.
            window: ``[start, end)``, slots from the start of the period, where the break lies.
            length: Consecutive free slots the break needs.
            where: Per element, whether it takes part; ``None``: every member.
        """
        rule = BreakRule(name, groups, interval, where, period, list(window), length)
        first, last = rule.window
        runs = [list(range(start, start + length)) for start in range(first, last - length + 1)]

        for group in self._groups(groups):
            members = group.where(where) if where is not None else group
            worker, occupant = members.member, members.member

            for index in range(interval.high // period):
                has_break = self.exists([
                    self.forall([~self._busy(rule, occupant, index, slot) for slot in run])
                    for run in runs
                ])
                works = self.count(worker, interval[worker] // period == index) > 0
                self.constraint(~works | has_break, name=name)

        self.break_rules.append(rule)

    def _busy(self, rule: "BreakRule", occupant: Member[Any], index: int, slot: int) -> Expr:
        """Whether a member that starts in period ``index`` covers its ``slot``.

        Args:
            rule: The breaks: their interval decision and period.
            occupant: A free member of the group.
            index: The period.
            slot: The slot, from the start of the period.

        Returns:
            At least one member starting in the period covers it (one reaching in does not).
        """
        interval, period = rule.interval, rule.period
        at = index * period + slot
        covers = (
            (interval[occupant] // period == index)
            & (interval[occupant] <= at)
            & (at < interval.end(occupant))
        )
        return self.count(occupant, covers) > 0

    def cumulative(
        self,
        groups: SubsetFamily[Any] | Assignment[Any] | Collection[Any],
        interval: IntegerDecision,
        demand: int | Parameter[Any],
        capacity: int | Parameter[Any],
        *,
        name: str,
    ) -> None:
        """A group's running intervals never use more than its capacity (CP-SAT ``AddCumulative``).

        Args:
            groups: A family (a group per subset), an assignment (one per target), or a collection.
            interval: The interval decision.
            demand: What a running interval uses of the capacity: one number, or per element.
            capacity: What a group offers at any moment: one number, or per assignment target.
            name: The constraint the violations are counted under.

        Raises:
            ValueError: A capacity per target on groups that are no assignment's targets.
        """
        if isinstance(capacity, Parameter) and not isinstance(groups, Assignment):
            msg = "a capacity per target needs groups=<an assignment>; give one number otherwise"
            raise ValueError(msg)  # noqa: TRY004  a model the engine refuses, not a wrong type

        # the load peaks at some interval's start, so checking every start checks every moment
        for group in self._groups(groups):
            member, other = group.member, group.member
            covers = (interval[other] <= interval[member]) & (
                interval[member] < interval.end(other)
            )
            used = demand[other] if isinstance(demand, Parameter) else demand
            load = self.sum(other, self.if_then_else(covers, used, 0))

            limit = (
                capacity[cast("AssignedGroup[Any]", group).target]
                if isinstance(capacity, Parameter)
                else capacity
            )
            self.constraint(load <= limit, name=name)

        self.resources.append(Resource(name, groups, interval, demand, capacity))

    def bin_packing(
        self,
        assignment: Assignment[Any],
        demand: int | Parameter[Any],
        capacity: int | Parameter[Any],
        *,
        name: str,
    ) -> None:
        """What a target holds never exceeds its capacity (CP Optimizer ``pack``), no time.

        Args:
            assignment: The assignment: one bin per target.
            demand: What an element uses of its target: one number, or per element.
            capacity: What a target holds: one number, or per target.
            name: The constraint the violations are counted under.
        """
        for group in assignment.groups:
            member = group.member
            load = self.sum(member, demand[member] if isinstance(demand, Parameter) else demand)
            limit = capacity[group.target] if isinstance(capacity, Parameter) else capacity
            self.constraint(load <= limit, name=name)

        self.resources.append(Resource(name, assignment, None, demand, capacity))

    @staticmethod
    def _groups(
        groups: SubsetFamily[Any] | Assignment[Any] | Collection[Any],
    ) -> list[Collection[Any]]:
        """The groups a primitive is posted on.

        Args:
            groups: A family, an assignment, or a collection.

        Returns:
            One group looping over the subsets or the targets, or the collection itself.
        """
        if isinstance(groups, Assignment):
            return list(groups.groups)

        if isinstance(groups, SubsetFamily):
            return list(groups)

        return [groups]

    def constraint(
        self, expr: Expr, *, name: str, scope: Collection[Any] | None = None, label: str = ""
    ) -> None:
        """A hard constraint: one violation under ``name`` each time ``expr`` is false.

        Args:
            expr: The condition.
            name: The constraint it is counted under.
            scope: A collection whose members' partial counts include it.
            label: Text for reports; ``{a}`` and ``{b}`` stand for the members involved.
        """
        term = Term(
            expr, name, violation=True, scope=[] if scope is None else [scope], label=label
        )
        self._add(term, None)

    def tie(self, name: str, how: Tie) -> None:
        """Declare how a search moves the elements the constraint ``name`` ties.

        Args:
            name: A constraint already posted.
            how: How the search moves the elements it ties.

        Raises:
            ValueError: No constraint has that name, or another one already ties that way.
        """
        if name not in self.weights:
            msg = f"{name!r} ties nothing: post the constraint before its tie"
            raise ValueError(msg)

        known = self.ties.setdefault(how, name)

        if known != name:
            msg = f"{known!r} and {name!r} are both {how}: one constraint per tie"
            raise ValueError(msg)

    def minimize(
        self,
        expr: Operand,
        *,
        name: str,
        weight: int,
        scope: Collection[Any] | None = None,
        label: str = "",
    ) -> None:
        """A soft term: ``expr`` added to the count kept under ``name``, weighed by ``weight``.

        Args:
            expr: The value.
            name: The term it is counted under.
            weight: What one unit of it costs in the soft score.
            scope: A collection whose members' partial counts include it.
            label: Text for reports.
        """
        term = Term(
            Expr.wrap(expr),
            name,
            violation=False,
            scope=[] if scope is None else [scope],
            label=label,
        )
        self._add(term, weight)

    def _add(self, term: Term, weight: int | None) -> None:
        """Keep a term and its constraint's weight.

        Args:
            term: The term.
            weight: Its weight; ``None`` for a hard constraint.

        Raises:
            ValueError: Its name was posted before as the other kind, or with another weight.
        """
        known = self.weights.setdefault(term.name, weight)

        if known != weight:
            kinds = [
                "hard" if each is None else f"soft, weight {each}" for each in (known, weight)
            ]
            msg = f"{term.name!r} is posted as {kinds[0]} and as {kinds[1]}: one name, one kind"
            raise ValueError(msg)

        self.terms.append(term)
