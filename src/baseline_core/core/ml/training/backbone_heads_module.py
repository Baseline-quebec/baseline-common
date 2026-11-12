"""Lightning module: a backbone and its heads, trained with the weighted sum of their losses."""

from typing import Any, ClassVar, cast, override

import pytorch_lightning as pl
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torchmetrics import MetricCollection

from baseline_core.core.ml.data.batch import AbstractBatch
from baseline_core.core.ml.data.phase import Phase
from baseline_core.core.ml.model.abstract_backbone import AbstractBackbone
from baseline_core.core.ml.model.head_container import HeadContainer
from baseline_core.core.ml.training.optimizer import OptimizerFactory


class StepOutput(BaseModel):
    """Result of one training, validation or test step.

    Attributes:
        loss: Scalar loss to optimize.
        losses: Loss components, logged as ``<phase>_<name>``.
        outputs: Predictions keyed by head target, for metrics and predictions.
    """

    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    outputs: dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BackboneHeadsModule(pl.LightningModule):
    """A backbone and its heads, trained with the weighted sum of the per-head losses.

    Attributes:
        backbone: Shared feature extractor.
        heads: Prediction heads, each with its loss and the inputs of its metrics.
        optimizer_factory: Builds the optimizer and scheduler in ``configure_optimizers``.
        metrics: Metric collections cloned per phase; read them through ``metrics_for``.
    """

    UNLOGGED_METRIC_SUFFIXES: ClassVar[tuple[str, ...]] = ("per_class", "classes")

    def __init__(
        self,
        backbone: AbstractBackbone[Any, Any],
        heads: HeadContainer,
        optimizer_factory: OptimizerFactory,
        metrics: dict[str, MetricCollection] | None = None,
    ) -> None:
        """Assemble the model and clone one metric collection per phase and target.

        Args:
            backbone: Shared feature extractor.
            heads: Prediction heads.
            optimizer_factory: Builds the optimizer and scheduler.
            metrics: Metric collections keyed by head target, or ``None`` for losses only.

        Raises:
            ValueError: If a metric collection names no head target.
        """
        super().__init__()
        metrics = metrics or {}
        unknown = sorted(set(metrics) - set(heads.targets))

        if unknown:
            error_msg = f"Metrics reference targets without a head: {unknown}"
            raise ValueError(error_msg)

        self.save_hyperparameters({
            "heads": {target: heads.head(target).config.model_dump() for target in heads.targets}
        })
        self.backbone = backbone
        self.heads = heads
        self.optimizer_factory = optimizer_factory
        self.metrics = nn.ModuleDict({
            self._metrics_key(phase): nn.ModuleDict({
                target: collection.clone(prefix=f"{phase.value}_{target}_")
                for target, collection in metrics.items()
            })
            for phase in Phase
        })

    def __call__(self, batch: AbstractBatch) -> dict[str, Any]:
        """Run ``forward`` through the ``nn.Module`` hooks, with a typed result."""
        preds: dict[str, Any] = super().__call__(batch)
        return preds

    @override
    def forward(self, batch: AbstractBatch) -> dict[str, Any]:
        """Run the backbone, then every head."""
        return self.heads(self.backbone(batch), batch)

    def step(self, batch: AbstractBatch) -> StepOutput:
        """Predict every head target and compute the weighted loss."""
        preds = self(batch)
        losses = self.heads.loss(preds, batch)
        return StepOutput(loss=losses["loss"], losses=losses, outputs=preds)

    def update_metrics(self, phase: Phase, output: StepOutput, batch: AbstractBatch) -> None:
        """Update the metrics of every head target with what its head hands them.

        Args:
            phase: Phase whose metrics are updated.
            output: Result of the step.
            batch: Batch the step ran on.
        """
        for target, collection in self.metrics_for(phase).items():
            collection.update(*self.heads.head(target).metric_args(output.outputs[target], batch))

    @override
    def on_fit_start(self) -> None:
        """Put the model in training mode, which ``from_pretrained`` and Lightning leave unset."""
        self.train()

    @override
    def training_step(self, batch: AbstractBatch, _batch_idx: int) -> torch.Tensor:
        """Step, logging and metrics for one training batch; only the loss is kept."""
        return self._run_step(batch, Phase.TRAIN).loss

    @override
    def validation_step(self, batch: AbstractBatch, _batch_idx: int) -> dict[str, Any]:
        """Step, logging and metrics for one validation batch, its output handed to callbacks."""
        return dict(self._run_step(batch, Phase.VAL))

    @override
    def test_step(self, batch: AbstractBatch, _batch_idx: int) -> dict[str, Any]:  # noqa: PT019
        """Step, logging and metrics for one test batch (a Lightning hook, not a pytest test)."""
        return dict(self._run_step(batch, Phase.TEST))

    @override
    def predict_step(self, batch: AbstractBatch, _batch_idx: int = 0) -> dict[str, Any]:
        """Predictions of every head, keyed by target."""
        return self(batch)

    @override
    def on_train_epoch_end(self) -> None:
        """Log the training metrics of the epoch."""
        self._log_epoch_metrics(Phase.TRAIN)

    @override
    def on_validation_epoch_end(self) -> None:
        """Log the validation metrics of the epoch."""
        self._log_epoch_metrics(Phase.VAL)

    @override
    def on_test_epoch_end(self) -> None:
        """Log the test metrics of the epoch."""
        self._log_epoch_metrics(Phase.TEST)

    @override
    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Build the optimizer and scheduler over the trainable parameters."""
        return self.optimizer_factory(param for param in self.parameters() if param.requires_grad)

    def metrics_for(self, phase: Phase) -> dict[str, MetricCollection]:
        """Metric collections of ``phase``, keyed by head target.

        Args:
            phase: Phase whose metrics are wanted.

        Returns:
            The collections registered for that phase; empty when there are none.
        """
        phase_metrics = cast("nn.ModuleDict", self.metrics[self._metrics_key(phase)])
        return {
            target: cast("MetricCollection", collection)
            for target, collection in phase_metrics.items()
        }

    @staticmethod
    def _metrics_key(phase: Phase) -> str:
        """Key of the metrics of ``phase`` in ``metrics``."""
        # Phase values alone cannot be ModuleDict keys: nn.Module already has a `train` method.
        return f"{phase.value}_metrics"

    def _run_step(self, batch: AbstractBatch, phase: Phase) -> StepOutput:
        """Run ``step``, log its losses per step and per epoch, and update the metrics."""
        output = self.step(batch)

        for name, value in output.losses.items():
            self.log(
                f"{phase.value}_{name}",
                value,
                prog_bar=name == "loss",
                on_epoch=True,
                batch_size=batch.batch_size,
                sync_dist=True,
            )

        self.update_metrics(phase, output, batch)
        return output

    def _log_epoch_metrics(self, phase: Phase) -> None:
        """Compute, log and reset the scalar metrics of ``phase`` updated during the epoch."""
        for collection in self.metrics_for(phase).values():
            if not any(metric.update_called for metric in collection.values()):
                continue

            values = {
                name: value
                for name, value in collection.compute().items()
                if value.numel() == 1 and not name.endswith(self.UNLOGGED_METRIC_SUFFIXES)
            }
            self.log_dict(values, sync_dist=True)
            collection.reset()
