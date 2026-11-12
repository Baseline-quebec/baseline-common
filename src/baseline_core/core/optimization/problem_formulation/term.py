"""One constraint or objective term of a model."""

from dataclasses import dataclass, field
from typing import Any

from baseline_core.core.optimization.problem_formulation.declaration import Collection
from baseline_core.core.optimization.problem_formulation.expression import Expr


@dataclass(slots=True)
class Term:
    """An expression, the constraint it is counted under, and how it counts."""

    # a dataclass, not a Pydantic model: binding creates one per instance, about a million at scale

    expr: Expr
    """The expression."""
    name: str
    """The constraint it belongs to: counts are kept per name."""
    violation: bool
    """``True``: a hard constraint, 1 each time ``expr`` is false; ``False``: a soft ``expr``."""
    scope: list[Collection[Any]] = field(default_factory=list)
    """Other elements whose partial counts include it: a collection, then positions once bound."""
    label: str = ""
    """Text for reports; ``{a}`` and ``{b}`` stand for the members involved."""
