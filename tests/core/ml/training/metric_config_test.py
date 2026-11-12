"""Tests for ``MetricConfig``."""

import pytest
import torchmetrics
from torchmetrics.classification import BinaryAccuracy

from baseline_core.core.ml.training.metric_config import MetricConfig


class TestMetricConfig:
    """Tests for ``MetricConfig``."""

    def test__build__when_top_level_name__should_instantiate_metric(self) -> None:
        """A bare class name is looked up in ``torchmetrics``."""
        metric = MetricConfig(name="MeanSquaredError").build()

        assert isinstance(metric, torchmetrics.MeanSquaredError)

    def test__build__when_dotted_name__should_look_up_submodule(self) -> None:
        """``submodule.Class`` is looked up in that torchmetrics submodule."""
        metric = MetricConfig(name="classification.BinaryAccuracy").build()

        assert isinstance(metric, BinaryAccuracy)

    def test__build__when_params__should_forward_them(self) -> None:
        """Constructor parameters are forwarded."""
        metric = MetricConfig(name="MeanSquaredError", params={"squared": False}).build()

        assert isinstance(metric, torchmetrics.MeanSquaredError)
        assert metric.squared is False

    def test__init__when_unknown_name__should_raise_value_error(self) -> None:
        """An unknown metric fails at validation."""
        with pytest.raises(ValueError, match="Metric 'NotAMetric' not found"):
            MetricConfig(name="NotAMetric")
