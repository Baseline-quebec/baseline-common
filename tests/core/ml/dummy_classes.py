"""Minimal implementations of the framework abstractions, shared by the ML tests."""

from typing import Any

import torch
from pydantic import BaseModel
from torch import nn

from baseline_core.core.ml.data.abstract_feature_loader import AbstractFeatureLoader
from baseline_core.core.ml.data.abstract_transform import AbstractTransform
from baseline_core.core.ml.data.batch import Batch, TensorTargets
from baseline_core.core.ml.model.abstract_backbone import AbstractBackbone
from baseline_core.core.ml.model.abstract_head import AbstractHead, HeadConfig, LossResult


class AddOne(AbstractTransform):
    """Add one to an integer."""

    def __call__(self, data: int) -> int:
        """Return ``data + 1``."""
        return data + 1


class Multiply(AbstractTransform):
    """Multiply an integer by a factor."""

    def __init__(self, factor: int) -> None:
        """Store the factor."""
        self.factor = factor

    def __call__(self, data: int) -> int:
        """Return ``data * factor``."""
        return data * self.factor


class ToText(AbstractTransform):
    """Convert anything to its string representation."""

    def __call__(self, data: Any) -> str:  # noqa: ANN401
        """Return ``str(data)``."""
        return str(data)


class SumValues(AbstractTransform):
    """Sum the values of a dict, as produced by a multi-key sample field."""

    def __call__(self, data: dict[str, int]) -> int:
        """Return the sum of the values."""
        return sum(data.values())


class UpperCaseLoader(AbstractFeatureLoader[str, str]):
    """Stands for a file loader: the stored value is a reference, the loaded value differs."""

    def load(self, stored: str) -> str:
        """Return ``stored`` in upper case."""
        return stored.upper()


class ScaleValues(AbstractTransform):
    """Multiply every value of a dict, keeping the keys: values transformed together."""

    def __init__(self, factor: int) -> None:
        """Store the factor."""
        self.factor = factor

    def __call__(self, data: dict[str, int]) -> dict[str, int]:
        """Return each value times ``factor``."""
        return {key: value * self.factor for key, value in data.items()}


class DummyBackboneConfig(BaseModel):
    """Backbone config with the fields the dummy backbone needs."""

    hidden_dim: int
    groups: list[str]


class DummyHeadConfig(HeadConfig):
    """Head config with the fields the dummy head needs."""

    group: str
    hidden_dim: int
    output_dim: int


class DummyBackbone(AbstractBackbone[Batch[TensorTargets], dict[str, torch.Tensor]]):
    """One linear layer per input."""

    def __init__(self, config: DummyBackboneConfig) -> None:
        """Create a linear layer for every group."""
        super().__init__()
        self.layers = nn.ModuleDict({
            group: nn.Linear(config.hidden_dim, config.hidden_dim) for group in config.groups
        })

    def forward(self, batch: Batch[TensorTargets]) -> dict[str, torch.Tensor]:
        """Project every group's features."""
        return {group: self.layers[group](x) for group, x in batch.x.items()}


class DummyHead(AbstractHead[Batch[TensorTargets], dict[str, torch.Tensor]]):
    """Linear projection of one group's embeddings, scored by an injected loss."""

    def __init__(self, config: DummyHeadConfig, loss_fn: nn.Module) -> None:
        """Create the projection and keep the loss."""
        super().__init__(config)
        self.group = config.group
        self.linear = nn.Linear(config.hidden_dim, config.output_dim)
        self.loss_fn = loss_fn

    def forward(
        self, features: dict[str, torch.Tensor], batch: Batch[TensorTargets]
    ) -> torch.Tensor:
        """Project the embeddings of ``config.group``."""
        del batch
        prediction: torch.Tensor = self.linear(features[self.group])
        return prediction

    def loss(
        self, predictions: torch.Tensor, batch: Batch[TensorTargets]
    ) -> torch.Tensor | LossResult:
        """``loss_fn`` of the predictions against the target of ``config.target``."""
        loss: torch.Tensor = self.loss_fn(predictions, self.truth(batch))
        return loss

    def metric_args(
        self, predictions: torch.Tensor, batch: Batch[TensorTargets]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The predictions and the target of ``config.target``."""
        return predictions, self.truth(batch)

    def truth(self, batch: Batch[TensorTargets]) -> torch.Tensor:
        """The target of ``config.target`` in ``batch``."""
        assert batch.targets is not None
        return batch.targets[self.config.target]


class LoggingHead(DummyHead):
    """Dummy head whose loss also reports the mean of its predictions."""

    def loss(self, predictions: torch.Tensor, batch: Batch[TensorTargets]) -> LossResult:
        """The dummy loss, plus ``pred_mean``."""
        loss: torch.Tensor = self.loss_fn(predictions, self.truth(batch))
        return LossResult.model_validate({"loss": loss, "pred_mean": predictions.mean()})
