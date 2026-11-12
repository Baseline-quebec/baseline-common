"""Expression trees: numbers and conditions combined by operators, as Python writes them."""

import itertools
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from numpy.typing import NDArray


type ArithmeticOperator = Literal["+", "-", "*", "//", "max", "min"]
type ComparisonOperator = Literal["<=", "<", "==", "!="]
type LogicalOperator = Literal["and", "or"]
type AggregateOperator = Literal["count", "sum", "min"]


class Expr:
    """A node of an expression tree; ``+``, ``<=``, ``&``… on nodes build larger nodes."""

    __slots__ = ()
    __hash__ = None  # type: ignore[assignment]

    SYMBOLS: ClassVar["itertools.count[int]"] = itertools.count()
    """Numbers free members and loops: two members of one collection are told apart by theirs."""

    @staticmethod
    def wrap(value: "Operand") -> "Expr":
        """Return ``value`` as a node, a number becoming a constant.

        Args:
            value: A node or a number.

        Returns:
            The node.
        """
        return value if isinstance(value, Expr) else Constant(float(value))

    @property
    def children(self) -> "list[Expr]":
        """The sub-expressions, in the order they are written."""
        return []

    def __add__(self, other: "Operand") -> "Expr":
        """The node ``self + other``."""
        return Arithmetic("+", self, Expr.wrap(other))

    def __radd__(self, other: "Operand") -> "Expr":
        """The node ``other + self``."""
        return Arithmetic("+", Expr.wrap(other), self)

    def __sub__(self, other: "Operand") -> "Expr":
        """The node ``self - other``."""
        return Arithmetic("-", self, Expr.wrap(other))

    def __rsub__(self, other: "Operand") -> "Expr":
        """The node ``other - self``."""
        return Arithmetic("-", Expr.wrap(other), self)

    def __mul__(self, other: "Operand") -> "Expr":
        """The node ``self * other``."""
        return Arithmetic("*", self, Expr.wrap(other))

    def __rmul__(self, other: "Operand") -> "Expr":
        """The node ``other * self``."""
        return Arithmetic("*", Expr.wrap(other), self)

    def __floordiv__(self, other: "Operand") -> "Expr":
        """The node ``self // other``."""
        return Arithmetic("//", self, Expr.wrap(other))

    def __neg__(self) -> "Expr":
        """The node ``0 - self``."""
        return Arithmetic("-", Constant(0.0), self)

    def __le__(self, other: "Operand") -> "Expr":
        """The condition ``self <= other``."""
        return Comparison("<=", self, Expr.wrap(other))

    def __ge__(self, other: "Operand") -> "Expr":
        """The condition ``other <= self``."""
        return Comparison("<=", Expr.wrap(other), self)

    def __lt__(self, other: "Operand") -> "Expr":
        """The condition ``self < other``."""
        return Comparison("<", self, Expr.wrap(other))

    def __gt__(self, other: "Operand") -> "Expr":
        """The condition ``other < self``."""
        return Comparison("<", Expr.wrap(other), self)

    def __eq__(self, other: object) -> "Expr":  # type: ignore[override]  # a node, as numpy does
        """The condition ``self == other`` (a node, not a truth value)."""
        return Comparison("==", self, Expr.wrap(other))  # type: ignore[arg-type]

    def __ne__(self, other: object) -> "Expr":  # type: ignore[override]  # a node, as numpy does
        """The condition ``self != other`` (a node, not a truth value)."""
        return Comparison("!=", self, Expr.wrap(other))  # type: ignore[arg-type]

    def __and__(self, other: "Operand") -> "Expr":
        """The condition ``self and other``."""
        return Logical("and", self, Expr.wrap(other))

    def __or__(self, other: "Operand") -> "Expr":
        """The condition ``self or other``."""
        return Logical("or", self, Expr.wrap(other))

    def __invert__(self) -> "Expr":
        """The condition ``not self``."""
        return Not(self)

    def __bool__(self) -> bool:
        """No truth value: ``if``, ``and``, ``or``, ``in`` would decide once, not per placement."""
        msg = "an expression has no truth value; combine it with &, |, ~ or m.if_then_else"
        raise TypeError(msg)


type Operand = float | Expr


@dataclass(eq=False, slots=True)
class Constant(Expr):
    """A number."""

    value: float


@dataclass(eq=False, slots=True)
class Lookup(Expr):
    """Data known once the model is bound, read at an index."""

    data: NDArray[Any]
    index: Expr

    @property
    def children(self) -> list[Expr]:
        """The index."""
        return [self.index]


@dataclass(eq=False, slots=True)
class Arithmetic(Expr):
    """Two numbers combined into one."""

    operator: ArithmeticOperator
    left: Expr
    right: Expr

    @property
    def children(self) -> list[Expr]:
        """Both operands."""
        return [self.left, self.right]


@dataclass(eq=False, slots=True)
class Comparison(Expr):
    """Two numbers compared: 1 when it holds, else 0."""

    operator: ComparisonOperator
    left: Expr
    right: Expr

    @property
    def children(self) -> list[Expr]:
        """Both operands."""
        return [self.left, self.right]


@dataclass(eq=False, slots=True)
class Logical(Expr):
    """Two conditions combined; the right one is evaluated only when it can change the result."""

    operator: LogicalOperator
    left: Expr
    right: Expr

    @property
    def children(self) -> list[Expr]:
        """Both conditions."""
        return [self.left, self.right]


@dataclass(eq=False, slots=True)
class Not(Expr):
    """A condition negated."""

    operand: Expr

    @property
    def children(self) -> list[Expr]:
        """The condition."""
        return [self.operand]


@dataclass(eq=False, slots=True)
class IfThenElse(Expr):
    """``when_true`` when the condition holds, else ``when_false``."""

    condition: Expr
    when_true: Expr
    when_false: Expr

    @property
    def children(self) -> list[Expr]:
        """The condition, then both branches."""
        return [self.condition, self.when_true, self.when_false]
