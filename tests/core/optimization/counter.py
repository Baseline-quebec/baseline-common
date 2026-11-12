"""Counts per constraint of a placement, by the kernel of a compiled model."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler


class Counter(BaseModel):
    """Counts per constraint of a placement: binds the model, loads the placement, counts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    compiled: ModelCompiler[Any]
    """The model whose constraints are counted."""

    def of(self, data: BaseModel, starts: list[int], targets: list[int]) -> dict[str, int]:
        """The count of every constraint for one placement.

        Args:
            data: The schema object.
            starts: Per element, its start.
            targets: Per element, the position of its target.

        Returns:
            The count, by constraint name.
        """
        kernel = self.compiled.bind(data).kernel
        kernel.set_placement(starts, targets)
        return dict(zip(self.compiled.model.names, kernel.full_counts, strict=True))
