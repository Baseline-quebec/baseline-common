"""Fit and test a training module with a Lightning trainer."""

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from loguru import logger
from pydantic import BaseModel, Field
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import Logger

from baseline_core.core.ml.training.backbone_heads_module import BackboneHeadsModule
from baseline_core.core.ml.training.callbacks import ThroughputMonitor
from baseline_core.core.ml.training.datamodule import AbstractDataModule


class TrainerConfig(BaseModel):
    """Lightning trainer settings.

    Attributes:
        max_epochs: Training epochs.
        accelerator: Lightning accelerator; ``"auto"`` uses a GPU when one is available.
        devices: Lightning devices.
        precision: Lightning precision.
        log_every_n_steps: Training steps between two metric logs.
        monitor_interval_s: Seconds between two training speed lines, or ``None`` for none.
    """

    max_epochs: int = Field(gt=0)
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str = "32-true"
    log_every_n_steps: int = Field(default=10, gt=0)
    monitor_interval_s: float | None = Field(default=30.0, gt=0)


class TrainingRun:
    """Fit or test a training module with an injected logger and callbacks.

    Attributes:
        trainer_config: Lightning trainer settings.
        experiment_logger: Lightning logger receiving metrics, or ``None`` for no logging.
        callbacks: Lightning callbacks, e.g. a ``ModelCheckpoint``.
    """

    def __init__(
        self,
        trainer_config: TrainerConfig,
        experiment_logger: Logger | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Store the trainer settings and the injected logger and callbacks.

        Args:
            trainer_config: Lightning trainer settings.
            experiment_logger: Lightning logger receiving metrics, or ``None`` for no logging.
            callbacks: Lightning callbacks; without a ``ModelCheckpoint`` nothing is saved.
        """
        self.trainer_config = trainer_config
        self.experiment_logger = experiment_logger
        self.callbacks = callbacks or []

    def fit(
        self,
        module: BackboneHeadsModule,
        datamodule: AbstractDataModule[Any],
        resume_from: Path | None = None,
    ) -> Path | None:
        """Train ``module``.

        Args:
            module: Module to train.
            datamodule: Training and validation data.
            resume_from: Checkpoint whose run is continued (optimizer state, epoch counter).

        Returns:
            The best checkpoint of the ``ModelCheckpoint`` callback, or ``None`` without one.
        """
        trainer = self._build_trainer()
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_from)
        checkpoint_callback = trainer.checkpoint_callback

        if not isinstance(checkpoint_callback, ModelCheckpoint):
            return None

        best_checkpoint = Path(
            checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        )
        logger.info("Best checkpoint: {}", best_checkpoint)
        return best_checkpoint

    def test(
        self, module: BackboneHeadsModule, datamodule: AbstractDataModule[Any]
    ) -> dict[str, float]:
        """Evaluate ``module`` on the test split.

        Args:
            module: Module to evaluate.
            datamodule: Test data.

        Returns:
            The test metrics logged by the module.
        """
        test_results = self._build_trainer().test(module, datamodule=datamodule)
        return {name: float(value) for name, value in test_results[0].items()}

    @staticmethod
    def load_weights(module: BackboneHeadsModule, checkpoint_path: Path) -> None:
        """Load the weights of a Lightning checkpoint into ``module``.

        Args:
            module: Module receiving the weights.
            checkpoint_path: Lightning checkpoint written by a run of this project.
        """
        # Our checkpoints also hold hyperparameters, which the weights-only loader rejects.
        checkpoint_content = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint_content["state_dict"])
        logger.info("Loaded weights from {}", checkpoint_path)

    def _build_trainer(self) -> pl.Trainer:
        """Build the Lightning trainer from ``trainer_config`` and the callbacks."""
        callbacks = list(self.callbacks)

        if self.trainer_config.monitor_interval_s is not None:
            callbacks.append(ThroughputMonitor(self.trainer_config.monitor_interval_s))

        return pl.Trainer(
            max_epochs=self.trainer_config.max_epochs,
            accelerator=self.trainer_config.accelerator,
            devices=self.trainer_config.devices,
            precision=self.trainer_config.precision,  # type: ignore[arg-type]
            logger=self.experiment_logger or False,
            callbacks=callbacks,
            enable_checkpointing=any(
                isinstance(callback, ModelCheckpoint) for callback in self.callbacks
            ),
            log_every_n_steps=self.trainer_config.log_every_n_steps,
        )
