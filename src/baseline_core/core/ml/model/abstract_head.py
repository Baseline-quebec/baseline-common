"""Heads: turn embeddings into predictions."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from baseline_core.core.ml.data.batch import AbstractBatch


class HeadConfig(BaseModel):
    """One prediction head; subclasses add the fields their head needs.

    Attributes:
        target: Key of this head's predictions and targets in the batch (e.g. ``"label"``).
        weight: Weight of this head's loss in the total loss.
    """

    target: str
    weight: float = Field(default=1.0, gt=0.0)

    model_config = ConfigDict(extra="allow")


class LossResult(BaseModel):
    """Loss plus intermediate values, as extra fields each logged under its own name.

    Attributes:
        loss: Scalar loss.
    """

    loss: torch.Tensor

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class AbstractHead[BatchT: AbstractBatch, FeaturesT](nn.Module, ABC):
    """Prediction head for one target.

    Attributes:
        config: Head configuration, including its target.
    """

    def __init__(self, config: HeadConfig) -> None:
        """Store the configuration.

        Args:
            config: Head configuration.
        """
        super().__init__()
        self.config = config

    def __call__(self, features: FeaturesT, batch: BatchT) -> Any:  # noqa: ANN401
        """Run ``forward`` through the ``nn.Module`` hooks."""
        return super().__call__(features, batch)

    @abstractmethod
    def forward(self, features: FeaturesT, batch: BatchT) -> Any:  # noqa: ANN401
        """Predict this head's target from the embeddings.

        Args:
            features: Embeddings from the backbone.
            batch: Batch the embeddings come from.

        Returns:
            Predictions for ``config.target``.
        """

    @abstractmethod
    def loss(self, predictions: Any, batch: BatchT) -> torch.Tensor | LossResult:  # noqa: ANN401
        """Score ``predictions`` against this head's supervision in ``batch``.

        Args:
            predictions: Output of ``forward``.
            batch: Batch the predictions come from.

        Returns:
            The scalar loss, or a ``LossResult`` with values worth logging.
        """

    @abstractmethod
    def metric_args(self, predictions: Any, batch: BatchT) -> tuple[Any, Any]:  # noqa: ANN401
        """Predictions and ground truth in the form this head's metrics expect.

        Args:
            predictions: Output of ``forward``.
            batch: Batch the predictions come from.

        Returns:
            ``(predictions, truth)`` handed to ``Metric.update``.
        """
