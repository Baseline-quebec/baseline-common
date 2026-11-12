"""Metrics named in configuration, resolved in ``torchmetrics``."""

from typing import Any

import torchmetrics
from pydantic import BaseModel, Field, field_validator

from baseline_core.core.ml.dynamic_import import DynamicImport


class MetricConfig(BaseModel):
    """A torchmetrics metric and its constructor parameters.

    Attributes:
        name: Class name in ``torchmetrics``, e.g. ``"detection.MeanAveragePrecision"``.
        params: Keyword arguments for the metric constructor.
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Fail fast when ``torchmetrics`` has no such metric."""
        cls._resolve(value)
        return value

    @classmethod
    def _resolve(cls, name: str) -> type[torchmetrics.Metric]:
        """Return the ``torchmetrics`` metric class named ``name``.

        Raises:
            ValueError: If ``torchmetrics`` has no such class.
            TypeError: If the class is not a ``torchmetrics.Metric``.
        """
        module_path, _, class_name = name.rpartition(".")
        module_name = f"torchmetrics.{module_path}" if module_path else "torchmetrics"

        try:
            metric_cls = DynamicImport.resolve_class(module_name, class_name)
        except ImportError as error:
            error_msg = f"Metric '{name}' not found in torchmetrics."
            raise ValueError(error_msg) from error

        if not issubclass(metric_cls, torchmetrics.Metric):
            error_msg = f"'{name}' is not a torchmetrics.Metric."
            raise TypeError(error_msg)

        return metric_cls

    def build(self) -> torchmetrics.Metric:
        """Instantiate the metric."""
        return self._resolve(self.name)(**self.params)
