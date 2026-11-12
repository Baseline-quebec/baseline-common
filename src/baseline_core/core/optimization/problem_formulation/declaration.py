"""What a model declares and what reads it: sets, parameters, decisions, members, schema paths."""

import functools
import typing
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, RootModel

from baseline_core.core.optimization.problem_formulation.expression import (
    AggregateOperator,
    Expr,
    Operand,
)


@dataclass(frozen=True, slots=True)
class RecordedPath:
    """Where a selector reads: the attributes taken from a set's element, or from the root."""

    schema: type[BaseModel] | None
    """The model type reached, or ``None`` past a plain value."""
    parts: list[str]
    """The attributes (and list positions) taken from ``origin``."""
    origin: "IndexSet[Any] | None"
    """The set whose elements the path starts on; ``None``: the schema root."""
    key: "RecordedPath | None" = None
    """When ``origin`` was reached by a key, ``jobs[op.job]``: the path to that key."""

    @property
    def name(self) -> str:
        """The path from the schema root, dotted: lists are crossed implicitly."""
        start = [] if self.origin is None else [self.origin.name]
        return ".".join([*start, *self.parts])

    @staticmethod
    def of(selected: object) -> "RecordedPath":
        """The path a selector returned: a ``SchemaPath``, or a list comprehension of one.

        Args:
            selected: What the selector returned.

        Returns:
            Its path.

        Raises:
            ValueError: The selector returned something else than a read of its argument.
        """
        if isinstance(selected, list) and len(selected) == 1:
            selected = selected[0]

        if not isinstance(selected, SchemaPath):
            msg = f"a selector returns a field of its argument, not {type(selected).__name__}"
            raise ValueError(msg)  # noqa: TRY004

        return selected._path  # noqa: SLF001

    @staticmethod
    def fields(schema: type[BaseModel]) -> list[str]:
        """The fields and public properties of a model type, sorted.

        Args:
            schema: The model type.

        Returns:
            Their names.
        """
        properties = {
            name
            for base in schema.__mro__
            if issubclass(base, BaseModel) and base not in {BaseModel, RootModel}
            for name, attribute in vars(base).items()
            if isinstance(attribute, property | functools.cached_property)
            and not name.startswith("_")
        }
        return sorted({*schema.model_fields, *properties})


class SchemaPath:
    """What a selector lambda receives: each attribute taken is checked against the schema."""

    __slots__ = ["_path"]

    def __init__(self, path: RecordedPath) -> None:
        """Stand at the end of a path.

        Args:
            path: The path recorded so far.
        """
        self._path = path

    def __getattr__(self, name: str) -> "SchemaPath":
        """The path one attribute further; a wrong name fails here, naming the valid ones."""
        if name.startswith("_"):
            raise AttributeError(name)

        path = self._path
        inner = self._unwrap(self._type_of(name))
        schema = inner if isinstance(inner, type) and issubclass(inner, BaseModel) else None
        return SchemaPath(RecordedPath(schema, [*path.parts, name], path.origin, path.key))

    def __getitem__(self, position: int) -> "SchemaPath":
        """One element of a list, by position: ``op.machines[0]``."""
        path = self._path
        return SchemaPath(
            RecordedPath(path.schema, [*path.parts, str(position)], path.origin, path.key)
        )

    def __iter__(self) -> Iterator["SchemaPath"]:
        """One pass: ``[option.resource_id for option in options]`` reads every option's id."""
        return iter([self])

    def _type_of(self, name: str) -> object:
        """A field's annotated type, or a property's return type (computed field included).

        Args:
            name: The attribute.

        Returns:
            Its type.

        Raises:
            ValueError: The schema has no such field or property.
        """
        schema = self._path.schema

        if schema is None:
            msg = f"{self._path.name!r} is a plain value: it has no field {name!r}"
            raise ValueError(msg)

        hints = typing.get_type_hints(schema)

        if name in hints:
            return hints[name]

        attribute = getattr(schema, name, None)

        if isinstance(attribute, property | functools.cached_property):
            getter = attribute.fget if isinstance(attribute, property) else attribute.func
            return typing.get_type_hints(getter).get("return")

        msg = f"{schema.__name__} has no field {name!r}; its fields: {RecordedPath.fields(schema)}"
        raise ValueError(msg)

    @staticmethod
    def _unwrap(inner: object) -> object:
        """``list[X]`` and a ``RootModel`` wrapping one both lead to ``X``.

        Args:
            inner: A type.

        Returns:
            The type of one element, for a list; the type itself otherwise.
        """
        while True:
            if isinstance(inner, type) and issubclass(inner, RootModel):
                inner = inner.model_fields["root"].annotation
            elif typing.get_origin(inner) is list:
                inner = typing.get_args(inner)[0]
            else:
                return inner


@dataclass(eq=False, slots=True)
class Member[T](Expr):
    """A free member of a collection, ranging over its members wherever it is read."""

    collection: "Collection[T]"
    symbol: int = field(default_factory=Expr.SYMBOLS.__next__)


@dataclass(eq=False, slots=True)
class Position(Expr):
    """The member at an offset from the current one in a sorted list (``-1``: the previous one)."""

    sorted_list: "SortedList"
    offset: int


@dataclass(eq=False, slots=True)
class Loop(Expr):
    """A loop over a set's elements or a family's subsets; binding replaces it by each value."""

    over: "IndexSet[Any] | SubsetFamily[Any]"
    symbol: int = field(default_factory=Expr.SYMBOLS.__next__)


@dataclass(eq=False, slots=True)
class ParameterRead(Expr):
    """A parameter read at one element; binding replaces it by the data it reads."""

    parameter: "Parameter[Any]"
    element: Expr

    @property
    def children(self) -> list[Expr]:
        """The element."""
        return [self.element]


@dataclass(eq=False, slots=True)
class ValueRead(Expr):
    """The integer decision of an element (its start)."""

    decision: "IntegerDecision"
    element: Expr

    @property
    def children(self) -> list[Expr]:
        """The element."""
        return [self.element]


@dataclass(eq=False, slots=True)
class TargetRead(Expr):
    """The target an element is assigned to."""

    assignment: "Assignment[Any]"
    element: Expr

    @property
    def children(self) -> list[Expr]:
        """The element."""
        return [self.element]


@dataclass(eq=False, slots=True)
class Size(Expr):
    """The number of members of a collection."""

    collection: "Collection[Any]"


@dataclass(eq=False, slots=True)
class Aggregate(Expr):
    """A value per member of a collection reduced to one number."""

    operator: AggregateOperator
    member: Member[Any]
    body: Expr

    @property
    def children(self) -> list[Expr]:
        """The member, then the value per member."""
        return [self.member, self.body]


class Collection[T]:
    """What has members to range over: ``group.member``, ``group.pair``."""

    @property
    def member(self) -> Member[T]:
        """A new free member: each access is another member, ranging over every member."""
        return Member(self)

    @property
    def pair(self) -> list[Member[T]]:
        """Two new free members, ranging over every pair of distinct members."""
        return [Member(self), Member(self)]

    @property
    def size(self) -> Expr:
        """The number of members."""
        return Size(self)

    def where(self, mask: "Parameter[Any]") -> "FilteredSet[T]":
        """Return the members for which a per-element parameter is non-zero.

        Args:
            mask: The per-element parameter.

        Returns:
            The filtered collection.
        """
        return FilteredSet(self, mask)


class FilteredSet[T](Collection[T]):
    """A collection restricted by a mask: the members a per-element parameter flags."""

    def __init__(self, base: Collection[T], mask: "Parameter[Any]") -> None:
        """Keep the base collection and the mask.

        Args:
            base: The collection filtered.
            mask: The per-element parameter; its data replaces it when the model is bound.
        """
        self.base = base
        self.mask = mask


class SortedList:
    """A collection ordered by an integer decision (ties by element position)."""

    def __init__(self, base: Collection[Any], key: "IntegerDecision") -> None:
        """Keep the collection and the sort key.

        Args:
            base: The collection sorted.
            key: The decision it is sorted by.
        """
        self.base = base
        self.key = key

    @property
    def current(self) -> Position:
        """The member at the current position: with ``previous``, every consecutive pair."""
        return Position(self, 0)

    @property
    def previous(self) -> Position:
        """The member just before the current one."""
        return Position(self, -1)


class IndexSet[T](Collection[T]):
    """Elements ``0 .. size - 1``, read from the schema; ``jobs[op.job]`` follows a key."""

    def __init__(
        self,
        name: str,
        schema: type[BaseModel] | None,
        key: str | None = None,
        *,
        distinct: bool = False,
    ) -> None:
        """Declare the set.

        Args:
            name: Where its elements are read (a schema path).
            schema: The model type of its elements; ``None`` for plain values.
            key: The field that names its elements, if any.
            distinct: Whether equal elements (copies) count once.

        Raises:
            ValueError: ``key`` is no field of the elements.
        """
        if key is not None and (schema is None or key not in RecordedPath.fields(schema)):
            fields = [] if schema is None else RecordedPath.fields(schema)
            msg = f"{name!r} is keyed by {key!r}, which is none of its fields: {fields}"
            raise ValueError(msg)

        self.name = name
        self.schema = schema
        self.key = key
        self.distinct = distinct

    def __getitem__(self, key: object) -> T:
        """Inside a selector, the element whose key is ``key``: the bind joins on it."""
        if self.key is None:
            msg = f"{self.name!r} has no key to follow: declare it with m.set(..., key=...)"
            raise ValueError(msg)

        if not isinstance(key, SchemaPath):
            msg = f"{self.name}[...] follows a key read inside a selector, not {key!r}"
            raise ValueError(msg)  # noqa: TRY004  a model the engine refuses, not a wrong type

        joined = RecordedPath(self.schema, [], self, RecordedPath.of(key))
        return cast("T", SchemaPath(joined))


@dataclass(frozen=True, slots=True)
class Through[B]:
    """A declaration read through another: ``base`` at the element ``index`` names."""

    base: B
    """The declaration read."""
    index: "Parameter[Any]"
    """Per element, the position in the base's set it reads."""


class Parameter[T]:
    """Integer data, one value per element of a set."""

    def __init__(
        self,
        name: str,
        over: IndexSet[T] | None,
        of: IndexSet[Any] | None = None,
        through: "Through[Parameter[Any]] | None" = None,
        values: list[int] | None = None,
    ) -> None:
        """Declare the parameter.

        Args:
            name: Where the values are read, or a name for values given in the model.
            over: The set it is indexed by; ``None`` for values given in the model.
            of: The set its values name, if any.
            through: The base it reads, at the element its index names, if any.
            values: The values themselves, when the model gives them (``m.lookup_table``).
        """
        self.name = name
        self.over = over
        self.of = of
        self.through = through
        self.values = values

    def __getitem__(self, element: Operand) -> Expr:
        """The parameter read at ``element``."""
        return ParameterRead(self, Expr.wrap(element))


class SubsetFamily[T]:
    """A list of subsets (of elements or plain values); ``for group in family`` loops over them."""

    def __init__(
        self,
        name: str,
        of: IndexSet[T] | None = None,
        *,
        through: "Through[SubsetFamily[T]] | None" = None,
        by: str | None = None,
        same_order_as: "SubsetFamily[Any] | None" = None,
    ) -> None:
        """Declare the family: read at ``name``, or one subset of ``of`` per key read at ``by``.

        Args:
            name: Where its subsets are read (a schema path), or a name for a grouping.
            of: The set its subsets draw from; ``None`` for plain values.
            through: The base it reads, at the element its index names, if any.
            by: For a grouping, the attribute holding an element's key, or its list of keys.
            same_order_as: For a grouping, another grouping whose order of keys to follow.
        """
        self.name = name
        self.of = of
        self.by = by
        self.same_order_as = same_order_as
        self.through = through

    def __iter__(self) -> "Iterator[Subset[T]]":
        """One loop variable: the loop body is written once and applies to every subset."""
        return iter([Subset(self, Loop(self))])

    def __getitem__(self, key: Operand) -> "Subset[T]":
        """The subset at ``key``."""
        return Subset(self, Expr.wrap(key))


class Subset[T](Collection[T]):
    """A family's subset selected by an expression (a loop over its subsets, or an element's)."""

    def __init__(self, family: SubsetFamily[T], key: Expr) -> None:
        """Keep the family and the selecting expression.

        Args:
            family: The family.
            key: Which subset.
        """
        self.family = family
        self.key = key


@dataclass(eq=False, slots=True)
class ListMember[T](Member[T]):
    """A member of a list read on an element (``m.within``): ``window.start`` reads its field."""

    items: IndexSet[T] = field(kw_only=True)
    """Every item of those lists, as one set: the fields are read per item."""
    reads: "dict[str, Parameter[T]]" = field(kw_only=True, default_factory=dict)
    """The fields read so far, one parameter each."""

    def __getattr__(self, name: str) -> Expr:
        """The field ``name`` read at this member; a wrong name fails here."""
        if name.startswith("_"):
            raise AttributeError(name)

        if name not in self.reads:
            on_item = SchemaPath(RecordedPath(self.items.schema, [], self.items))
            path = RecordedPath.of(getattr(on_item, name))
            self.reads[name] = Parameter(path.name, self.items)

        return ParameterRead(self.reads[name], self)


class Assignment[T]:
    """Decision: each ``of`` element goes to one of ``to``; ``assignment[element]`` reads it."""

    def __init__(
        self, name: str, of: IndexSet[T], to: IndexSet[Any], among: SubsetFamily[Any] | None = None
    ) -> None:
        """Declare the decision.

        Args:
            name: Its name, by which a schedule is loaded.
            of: The elements assigned.
            to: The targets.
            among: Per element of ``of``, the targets it may go to.
        """
        self.name = name
        self.of = of
        self.to = to
        self.among = among

    def __getitem__(self, element: Operand) -> Expr:
        """The target ``element`` is assigned to."""
        return TargetRead(self, Expr.wrap(element))

    @property
    def groups(self) -> "list[AssignedGroup[T]]":
        """One group, looping over the targets: the elements sent to that target."""
        return [AssignedGroup(self, Loop(self.to))]


class AssignedGroup[T](Collection[T]):
    """The elements assigned to one target, inside ``for group in assignment.groups``."""

    def __init__(self, assignment: Assignment[T], target: Loop) -> None:
        """Keep the decision and the loop over its targets.

        Args:
            assignment: The decision.
            target: The loop over its targets: the target this group gathers.
        """
        self.assignment = assignment
        self.target = target


class IntegerDecision:
    """Decision: an integer per set element, optionally an interval ``[start, start + span)``."""

    def __init__(
        self,
        name: str,
        over: IndexSet[Any],
        span: Parameter[Any] | None,
        low: int,
        high: int,
        period: int | None = None,
    ) -> None:
        """Declare the decision.

        Args:
            name: Its name, by which a schedule is loaded.
            over: The elements.
            span: Per element, how long it lasts; makes the decision an interval.
            low: Least value.
            high: Greatest end: the search and the pre-check list every value up to it.
            period: Values per period, if the values come in periods.
        """
        self.name = name
        self.over = over
        self.span = span
        self.low = low
        self.high = high
        self.period = period

    def __getitem__(self, element: Operand) -> Expr:
        """The decision of ``element``."""
        return ValueRead(self, Expr.wrap(element))

    def end(self, element: Operand) -> Expr:
        """Where the interval of ``element`` ends.

        Args:
            element: A member or a position.

        Returns:
            Its start plus its span.

        Raises:
            ValueError: The decision has no span.
        """
        if self.span is None:
            msg = f"{self.name!r} has no span; declare it with span="
            raise ValueError(msg)

        return self[element] + self.span[element]
