from typing import Literal

import numpy as np
import pytest
from pydantic import BaseModel

from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.model import Model
from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler


class Gauge(BaseModel):
    ref: str
    level: float


class Panel(BaseModel):
    gauges: list[Gauge]


type Threshold = Literal["literal", "data", "weight"]


class ThresholdDefinition(AbstractModelDefinition[Panel]):
    """One rule on the first gauge's start, its number written in the model or read in data."""

    schema = Panel

    def __init__(self, threshold: Threshold) -> None:
        """Choose where the rule's number comes from.

        Args:
            threshold: Written as a literal, read on the gauge, or a factor of a soft term.
        """
        self.threshold = threshold

    def define(self, model: Model[Panel]) -> None:
        """Post the rule.

        Args:
            model: The model to post on.
        """
        gauges = model.set(lambda s: s.gauges, key="ref")
        level = model.param(lambda gauge: gauge.level, gauges)
        at = model.var("at", gauges, high=5)

        match self.threshold:
            case "literal":
                model.constraint(at[0] >= 2.5, name="threshold")
            case "data":
                model.constraint(at[0] >= level[0], name="threshold")
            case "weight":
                model.minimize(at[0] * 0.5, name="threshold", weight=1)


class TestTemplateConstants:
    @pytest.mark.parametrize("cythonize", [False, True])
    @pytest.mark.parametrize("threshold", ["literal", "data", "weight"])
    def test__a_fractional_number__is_refused_at_bind(
        self, *, threshold: Threshold, cythonize: bool
    ) -> None:
        """A number the kernel would truncate to an integer is refused, never counted wrong."""
        panel = Panel(gauges=[Gauge(ref="g", level=2.5)])
        compiled = ModelCompiler(ThresholdDefinition(threshold), cythonize=cythonize)

        with pytest.raises(ValueError, match=r"'threshold'.*integers.*\d\.5"):
            compiled.bind(panel)

    @pytest.mark.parametrize("cythonize", [False, True])
    def test__a_whole_number_read_as_a_float__is_counted_as_written(
        self, *, cythonize: bool
    ) -> None:
        """A level of 2.0 is the integer 2: a start at 1 breaks the rule, a start at 2 keeps it."""
        panel = Panel(gauges=[Gauge(ref="g", level=2.0)])
        kernel = ModelCompiler(ThresholdDefinition("data"), cythonize=cythonize).bind(panel).kernel
        counts = []

        for start in [1, 2]:
            kernel.load(at=np.array([start], dtype=np.int64))
            counts.append(kernel.full_hard)

        assert counts == [1, 0]
