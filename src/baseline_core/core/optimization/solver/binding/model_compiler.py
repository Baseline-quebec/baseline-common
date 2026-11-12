"""A model written once and compiled: each ``bind`` fills a kernel with one set of data."""

import gc
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel

from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.model import Model
from baseline_core.core.optimization.solver.binding.binder import Binder, BoundData
from baseline_core.core.optimization.solver.kernel.kernel import KernelLayout
from baseline_core.core.optimization.solver.kernel.kernel_builder import KernelBuilder
from baseline_core.core.optimization.solver.kernel.kernel_source import KernelSource
from baseline_core.core.optimization.solver.kernel.shape import BoundStatement, Template
from baseline_core.core.optimization.solver.model_solution import (
    ModelSolution,
    Summary,
    TermSummary,
    VariableAssignment,
)
from baseline_core.core.optimization.solver.score import Score


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from baseline_core.core.optimization.solver.kernel.kernel import Kernel
    from baseline_core.core.optimization.solver.kernel.model_kernel import ModelKernel


@dataclass(frozen=True, slots=True)
class BoundModel:
    """What ``ModelCompiler.bind`` returns: the kernel and what was read to make it."""

    kernel: "Kernel"
    """The generated kernel over the data, every element at 0 in set 0."""
    data: BoundData
    """The problem's data as the model reads it: elements, targets, values, subsets."""
    model: Model[Any]
    """The model bound: its declarations, decisions and weights."""
    templates: list[Template]
    """The templates the kernel was generated from, one per shape of term, for the pre-check."""
    element_positions: dict[str, int]
    """Each element's position, by its key; by its position as text when the set declares none."""
    target_sets: dict[str, int]
    """Each target's set, by its key; empty without an assignment."""

    def __post_init__(self) -> None:
        """Refuse keys that name two elements or two targets.

        Raises:
            ValueError: Two elements, or two targets, share a key.
        """
        shared_element = len(self.element_positions) < len(self.data.elements)
        assigned = self.model.assignment is not None

        if shared_element or (assigned and len(self.target_sets) < len(self.data.targets)):
            msg = "two elements, or two targets, share a key"
            raise ValueError(msg)

    def place(self, warm_start: list[VariableAssignment]) -> None:
        """Load every element at its lowest value in its first allowed target, then the warm start.

        Args:
            warm_start: Where the search begins; an assignment the model cannot read is skipped.
        """
        interval, assign = self.model.integer_decision, self.model.assignment
        elements = self.data.elements
        low = 0 if interval is None else interval.low
        allowed = (
            [[0]] * len(elements)
            if assign is None or assign.among is None
            else self.data.subsets[assign.among]
        )

        value_name, set_name = self.kernel.variables
        arrays: dict[str, list[int]] = {}

        if value_name is not None:
            arrays[value_name] = [low] * len(elements)

        if set_name is not None:
            arrays[set_name] = [row[0] if row else 0 for row in allowed]

        self.kernel.load(**arrays)

        if warm_start:
            self._fold(warm_start)

    def _fold(self, warm_start: list[VariableAssignment]) -> None:
        """Move each element the warm start names to its value and target, the rest as they are.

        Args:
            warm_start: The assignments; one the model cannot read is skipped.
        """
        starts, chosen = self.kernel.placement

        for assignment in warm_start:
            if self.unreadable_reason(assignment) is not None:
                continue

            element = self.element_positions[assignment.element]
            target = assignment.target
            self.kernel.insert(
                element,
                starts[element] if assignment.value is None else assignment.value,
                chosen[element] if target is None else self.target_sets[target],
            )

    def unreadable_reason(self, assignment: VariableAssignment) -> str | None:
        """Why the model cannot read an assignment; ``None`` when it can.

        Args:
            assignment: The assignment.

        Returns:
            The reason, or ``None``.
        """
        element = self.element_positions.get(assignment.element)

        if element is None:
            return f"{assignment.element!r} is no element of the model"

        if assignment.target is not None and assignment.target not in self.target_sets:
            return f"{assignment.target!r} is no target of the model"

        value, interval = assignment.value, self.model.integer_decision

        if value is None:
            return None

        if interval is None:
            return f"{assignment.element!r} has no value: the model declares no integer variable"

        high = interval.high

        if value < interval.low or value + self.span(element) > high:
            return f"{assignment.element!r} at {value} leaves [{interval.low}, {high}]"

        return None

    def span(self, element: int) -> int:
        """How long an element lasts: 0 without a span.

        Args:
            element: The element's position.

        Returns:
            Its span.
        """
        interval = self.model.integer_decision

        if interval is None or interval.span is None:
            return 0

        return int(self.data.values[interval.span][element])

    @property
    def solution(self) -> ModelSolution:
        """The kernel's assignments of the present elements, the absent ones, and the summary."""
        interval, assign = self.model.integer_decision, self.model.assignment
        values, chosen = self.kernel.placement
        absent = self.kernel.absent_mask
        element_keys, target_keys = list(self.element_positions), list(self.target_sets)

        assignments = [
            VariableAssignment(
                element=key,
                value=None if interval is None else int(values[position]),
                target=None if assign is None else target_keys[chosen[position]],
            )
            for position, key in enumerate(element_keys)
            if not absent[position]
        ]

        weights = self.model.weights
        # plain ints: run as pure Python, the kernel counts in NumPy integers, which JSON refuses
        count = {
            name: int(value)
            for name, value in zip(self.kernel.names, self.kernel.full_counts, strict=True)
        }
        summary = Summary(
            score=Score(hard=int(self.kernel.full_hard), soft=int(self.kernel.full_soft)),
            terms={
                name: TermSummary(count[name], weight, count[name] * weight)
                for name, weight in weights.items()
                if weight is not None
            },
            constraints={name: count[name] for name, weight in weights.items() if weight is None},
        )

        return ModelSolution(
            assignments=assignments,
            unassigned=[key for position, key in enumerate(element_keys) if absent[position]],
            summary=summary,
        )


class ModelCompiler[S: BaseModel]:
    """A model ready to bind; its kernel source depends on the model alone, never on the data."""

    def __init__(self, definition: AbstractModelDefinition[S], *, cythonize: bool = True) -> None:
        """Write the model from its definition (``define`` runs once, here).

        Args:
            definition: The definition to write the model from.
            cythonize: Whether to compile kernels with Cython; ``False`` runs them as plain Python.
        """
        self.model = Model(definition.schema)
        definition.define(self.model)

        self.builder = KernelBuilder(cythonize=cythonize)
        self.binder = Binder(self.model)

    def bind(self, problem: S) -> BoundModel:
        """The model over ``problem``: a kernel, and the problem's data as the model reads it.

        Args:
            problem: The schema object to read every declaration from.

        Returns:
            The bound model, its kernel's every element at 0 in set 0.
        """
        model = self.model

        with self._paused_collector():
            data = self.binder.expand(problem)
            templates = self._templates(data.statements)

            setlevel = np.zeros(len(model.names), dtype=np.int64)

            for template in templates:
                if template.set_level:
                    setlevel[template.name_index] = 1

            n_elements, n_sets = len(data.elements), max(len(data.targets), 1)
            ranks = self._ranks(data.elements)
            generated = KernelSource.write(templates, n_elements, n_sets, ranks)

            kernel_class: type[ModelKernel] = self.builder.load(generated.source).ModelKernel
            layout = KernelLayout(
                n=n_elements,
                n_sets=n_sets,
                names=model.names,
                variables=model.variables,
                labels=[template.labels for template in templates],
                objects=data.elements,
                setlevel=setlevel,
                weights=model.unit_weights,
                hard=model.hard,
            )
            kernel = kernel_class(generated.arrays, layout)

        decided, assign = model.decided_set, model.assignment
        return BoundModel(
            kernel,
            data,
            model,
            templates,
            self._positions(data.elements, None if decided is None else decided.key),
            {} if assign is None else self._positions(data.targets, assign.to.key),
        )

    @staticmethod
    def _positions(items: list[Any], field: str | None) -> dict[str, int]:
        """Each item's position, by the field its set declares as text, else by its position.

        Args:
            items: The schema objects, in position order.
            field: The field the set declares as its key, if any.

        Returns:
            The positions, by key.
        """
        if field is None:
            return {str(position): position for position in range(len(items))}

        # keys travel as text (JSON): an int field keys as its digits
        return {str(getattr(item, field)): position for position, item in enumerate(items)}

    def _ranks(self, elements: list[Any]) -> "NDArray[np.int64]":
        """Each element's rank in its key's order, its position when the set declares no key.

        Args:
            elements: The schema objects, in position order.

        Returns:
            Per element, its rank: how a sorted list breaks ties.
        """
        over = self.model.decided_set
        ranks = np.arange(max(len(elements), 1), dtype=np.int64)

        if over is None or over.key is None or not elements:
            return ranks

        keys = [getattr(element, over.key) for element in elements]
        ranks[sorted(range(len(elements)), key=keys.__getitem__)] = np.arange(len(elements))
        return ranks

    @staticmethod
    @contextmanager
    def _paused_collector() -> Generator[None]:
        """Pause Python's garbage collector inside the block, and restore it after, error or not.

        Yields:
            Nothing: the block runs with the collector paused.
        """
        was_enabled = gc.isenabled()
        gc.disable()

        try:
            yield
        finally:
            if was_enabled:
                gc.enable()

    def _templates(self, statements: list[BoundStatement]) -> list[Template]:
        """Group the bound statements by shape: one template per shape under one constraint.

        Args:
            statements: Every statement, bound.

        Returns:
            The templates.
        """
        index = {name: position for position, name in enumerate(self.model.names)}
        groups: dict[str, list[BoundStatement]] = {}

        for statement in statements:
            kind = "V" if statement.probe.violation else "A"
            key = f"{index[statement.probe.name]}|{kind}|{statement.shape.key}"
            groups.setdefault(key, []).append(statement)

        return [
            Template(position, index[group[0].probe.name], group)
            for position, group in enumerate(groups.values())
        ]
