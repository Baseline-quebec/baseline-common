"""The values of a model's declarations, read from one schema object."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum
from operator import attrgetter, itemgetter
from typing import Any

from pydantic import BaseModel, RootModel

from baseline_core.core.optimization.problem_formulation.declaration import (
    IndexSet,
    Parameter,
    SubsetFamily,
)


@dataclass(frozen=True, slots=True)
class Nested:
    """A value read at a schema path, nested once per list crossed on the way."""

    value: Any
    """The value: a leaf, or lists of leaves."""
    levels: int
    """How many lists deep the leaves are."""


@dataclass(frozen=True, slots=True)
class ElementIndex:
    """The elements of one set, and how to find an element's position."""

    elements: list[Any]
    """The elements, in order."""
    by_identity: dict[int, int]
    """Position by ``id()`` of the object (copies of a distinct element point to the one kept)."""
    by_key: dict[Hashable, int]
    """Position by the value of the set's ``key`` attribute."""


class SchemaReader:
    """``operations.duration``: one value per operation; a text names an element or a category."""

    def __init__(self, root: BaseModel) -> None:
        """Read from one schema object; what is read is kept, so each set is indexed once.

        Args:
            root: The schema object.
        """
        self.root = root
        self._reads: dict[int, Any] = {}
        # two sets on one path differ by their key and distinct: each reads its own index
        self._sets: dict[tuple[str, str | None, bool], ElementIndex] = {}
        self._codes: dict[str, dict[str, int]] = {}
        self._keys: dict[int, list[Hashable]] = {}

    def read(self, declaration: IndexSet[Any] | Parameter[Any] | SubsetFamily[Any]) -> Any:  # noqa: ANN401  values of the user's schema
        """The data of one declaration, read once.

        Args:
            declaration: A set, a parameter or a family.

        Returns:
            A size, a list of values, or a list of subsets.
        """
        if id(declaration) not in self._reads:
            if isinstance(declaration, IndexSet):
                value: Any = len(self._index(declaration).elements)
            elif isinstance(declaration, SubsetFamily):
                grouped = declaration.by is not None
                value = self._grouped(declaration) if grouped else self._family(declaration)
            else:
                value = self._param(declaration)

            self._reads[id(declaration)] = value

        return self._reads[id(declaration)]

    def keys(self, family: SubsetFamily[Any]) -> list[Hashable] | None:
        """The key of each subset of a grouping, in its subsets' order.

        Args:
            family: The family.

        Returns:
            The keys, or ``None`` for a family read as it is in the schema.
        """
        if family.by is None:
            return None

        self.read(family)
        return self._keys[id(family)]

    def elements(self, index_set: IndexSet[Any]) -> list[Any]:
        """The objects of a set, in their positions' order.

        Args:
            index_set: The set.

        Returns:
            The objects.
        """
        return self._index(index_set).elements

    def _index(self, index_set: IndexSet[Any]) -> ElementIndex:
        """The elements of a set and their positions, read once.

        Args:
            index_set: The set.

        Returns:
            Its index.
        """
        cache = (index_set.name, index_set.key, index_set.distinct)

        if cache not in self._sets:
            walked = self._walk(index_set.name)
            read = self._flatten(walked.value, walked.levels)
            elements = read

            if index_set.distinct:
                elements = list(
                    {self._identity(element): element for element in reversed(read)}.values()
                )[::-1]

            by_identity = {
                id(element): position for position, element in reversed(list(enumerate(elements)))
            }

            if index_set.distinct:
                kept = {
                    self._identity(element): position for position, element in enumerate(elements)
                }
                by_identity |= {id(element): kept[self._identity(element)] for element in read}

            by_key: dict[Hashable, int] = {}

            if index_set.key:
                by_key = {
                    getattr(element, index_set.key): position
                    for position, element in reversed(list(enumerate(elements)))
                }

            self._sets[cache] = ElementIndex(
                elements=elements, by_identity=by_identity, by_key=by_key
            )

        return self._sets[cache]

    @staticmethod
    def _identity(value: object) -> str:
        """Equal values give the same text, so ``distinct`` keeps one of them.

        Args:
            value: An element.

        Returns:
            Its text.
        """
        return value.model_dump_json() if isinstance(value, BaseModel) else repr(value)

    def _param(self, param: Parameter[Any]) -> int | list[int]:
        """The values of a parameter read from the schema object.

        Args:
            param: The parameter.

        Returns:
            One value, or one per element.
        """
        over = param.over

        if over is not None and param.name.startswith(over.name + "."):
            return [self._value(value, param) for value in self._per_element(param.name, over)]

        walked = self._walk(param.name)

        if walked.levels == 0:
            return self._value(walked.value, param)

        return [self._value(item, param) for item in self._flatten(walked.value, walked.levels)]

    def _family(self, family: SubsetFamily[Any]) -> list[list[int]]:
        """The subsets of a family read from the schema object.

        Args:
            family: The family.

        Returns:
            One list of values (or positions) per subset.
        """
        walked = self._walk(family.name)
        rows = self._flatten(walked.value, max(walked.levels - 1, 0))

        if family.of is not None:
            by_identity = self._index(family.of).by_identity

            while any(
                isinstance(item, list) and id(item) not in by_identity
                for row in rows
                for item in row
            ):
                rows = [item for row in rows for item in row]

        return [[self._value(item, family) for item in row] for row in rows]

    def _grouped(self, family: SubsetFamily[Any]) -> list[list[int]]:
        """One subset of ``family.of`` per key found at ``family.by``, in the keys' order.

        Args:
            family: A family declared with ``m.group_by``.

        Returns:
            One list of element positions per key.

        Raises:
            ValueError: The family groups no set.
        """
        of, by = family.of, family.by

        if of is None or by is None:
            msg = f"{family.name!r} is no grouping of a set"
            raise ValueError(msg)

        members: dict[Hashable, list[int]] = {}

        for element, value in enumerate(self._per_element(by, of)):
            for key in value if isinstance(value, list) else [value]:
                if key not in {None, ""}:
                    members.setdefault(key, []).append(element)

        if family.same_order_as is not None:
            self.read(family.same_order_as)
            ordered = self._keys[id(family.same_order_as)]
        else:
            ordered = sorted(members, key=str)

        self._keys[id(family)] = ordered
        return [sorted(set(members.get(key, []))) for key in ordered]

    def _per_element(self, path: str, index_set: IndexSet[Any]) -> list[Any]:
        """The value at ``path`` for each element of a set; ``path`` starts where the set does.

        Args:
            path: The schema path.
            index_set: The set.

        Returns:
            One value per element.

        Raises:
            ValueError: The path is not read on the set's elements.
        """
        if not path.startswith(index_set.name + "."):
            msg = f"{path!r} is not read on the elements of {index_set.name!r}"
            raise ValueError(msg)

        parts = path[len(index_set.name) + 1 :].split(".")
        out = []

        for element in self._index(index_set).elements:
            value: Any = element

            for part in parts:
                value = value[int(part)] if part.isdigit() else self._unwrap(getattr(value, part))

            out.append(value)

        return out

    def _value(self, raw: object, declaration: Parameter[Any] | SubsetFamily[Any]) -> int:
        """A number: an ``of`` element becomes its position, a text without ``of`` a category.

        Args:
            raw: The value read.
            declaration: The parameter or family it was read for.

        Returns:
            The number.

        Raises:
            ValueError: The value names no element of ``of``, or cannot become a number.
        """
        if isinstance(raw, bool | int | float):
            # floats are refused later
            return int(raw) if isinstance(raw, bool) else raw  # type: ignore[return-value]

        if declaration.of is not None:
            index = self._index(declaration.of)
            is_key = isinstance(raw, str) and not isinstance(raw, Enum)
            found = index.by_key.get(raw) if is_key else index.by_identity.get(id(raw))

            if found is None:
                msg = f"{declaration.name!r}: {raw!r} is no element of {declaration.of.name!r}"
                raise ValueError(msg)

            return found

        if raw is None:
            return -1

        if isinstance(raw, Enum):
            return list(type(raw)).index(raw)

        if isinstance(raw, str):
            codes = self._codes.setdefault(declaration.name, {})
            return codes.setdefault(raw, len(codes))

        msg = f"{declaration.name!r} holds {type(raw).__name__}; say which set it names with of="
        raise ValueError(msg)

    def _walk(self, name: str) -> Nested:
        """The value at ``name``, nested once per list crossed on the way.

        Args:
            name: The schema path.

        Returns:
            The value, and how many lists deep it is.
        """
        value: Any = self.root
        levels = 0

        for part in name.split("."):
            if part.isdigit():
                levels -= 1
                value = self._map(value, levels, itemgetter(int(part)))
                continue

            value = self._map(value, levels, attrgetter(part))
            value = self._map(value, levels, self._unwrap)

            if all(isinstance(leaf, list) for leaf in self._flatten(value, levels)):
                levels += 1

        return Nested(value, levels)

    def _map(
        self,
        value: Any,  # noqa: ANN401  values of the user's schema
        depth: int,
        step: Callable[[Any], Any],
    ) -> Any:  # noqa: ANN401  values of the user's schema
        """Apply ``step`` to every leaf ``depth`` lists deep.

        Args:
            value: The nested lists.
            depth: How deep the leaves are.
            step: What to apply.

        Returns:
            The same nesting, with the leaves replaced.
        """
        return step(value) if depth == 0 else [self._map(item, depth - 1, step) for item in value]

    @staticmethod
    def _flatten(value: Any, levels: int) -> list[Any]:  # noqa: ANN401  user's schema values
        """The leaves of a value nested ``levels`` lists deep, in order.

        Args:
            value: The nested lists.
            levels: How deep.

        Returns:
            The leaves.
        """
        out = value if levels else [value]

        for _ in range(levels - 1):
            out = [item for row in out for item in row]

        return out

    @staticmethod
    def _unwrap(value: object) -> object:
        """A ``RootModel`` reads as the list it wraps; any sequence reads as a list.

        Args:
            value: The value read.

        Returns:
            The value, as a list where it is a sequence.
        """
        value = value.root if isinstance(value, RootModel) else value
        return list(value) if isinstance(value, tuple | frozenset) else value
