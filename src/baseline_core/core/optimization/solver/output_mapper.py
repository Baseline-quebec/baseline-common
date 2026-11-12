"""A problem's output, declared as a schema and filled from a solution: a configuration."""

from typing import Any

from pydantic import BaseModel

from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.model_solution import ModelSolution


class OutputMapper[Out: BaseModel]:
    """Fill a problem's output schema from a solution, one ``assigned`` entry per element."""

    def __init__(self, schema: type[Out]) -> None:
        """Keep the output schema the mapping reads.

        Args:
            schema: A model whose fields read, by ``AliasPath``, an ``assigned`` entry's keys.
        """
        self.schema = schema

    def map(self, bound: BoundModel, solution: ModelSolution) -> Out:
        """Fill the schema from ``assigned``: ``element``, ``value``, ``end``, ``target`` each.

        Args:
            bound: The model bound to the problem the solution was found on.
            solution: The solution, in keys.

        Returns:
            The output.
        """
        elements, targets = bound.data.elements, bound.data.targets
        assigned: list[dict[str, Any]] = []

        for assignment in solution.assignments:
            position, value, key = (
                bound.element_positions[assignment.element],
                assignment.value,
                assignment.target,
            )
            element = elements[position]
            target = None if key is None else targets[bound.target_sets[key]]
            # dumped: an alias path walks dicts all the way, never a dict then an object
            assigned.append({
                "element": element.model_dump() if isinstance(element, BaseModel) else element,
                "value": value,
                "end": None if value is None else value + bound.span(position),
                "target": target.model_dump() if isinstance(target, BaseModel) else target,
            })

        return self.schema.model_validate({"assigned": assigned})
