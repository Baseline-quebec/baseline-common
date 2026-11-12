"""Lightning callbacks for the training loop."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, override

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from baseline_core.core.ml.tracking.checkpoint import MLFlowCheckpointStore


class ThroughputMonitor(pl.Callback):
    """Report training speed and the share of step time spent waiting for batches."""

    def __init__(
        self,
        interval_s: float,
        emit: Callable[[str], None] = logger.info,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Prepare the counters.

        Args:
            interval_s: Seconds between two reports.
            emit: Where report lines go.
            clock: Time source, in seconds.
        """
        self._interval_s = interval_s
        self._emit = emit
        self._clock = clock
        self._window_start: float | None = None
        self._last_batch_end = 0.0
        self._batch_start = 0.0
        self._data_wait_s = 0.0
        self._compute_s = 0.0
        self._steps = 0

    @override
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Start measuring data wait from the epoch start, not from the last validation."""
        del trainer, pl_module
        self._last_batch_end = self._clock()

        if self._window_start is None:
            self._window_start = self._last_batch_end

    @override
    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Count the time since the previous batch ended as waiting for data."""
        del trainer, pl_module, batch, batch_idx
        self._batch_start = self._clock()
        self._data_wait_s += self._batch_start - self._last_batch_end

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Count the step as compute, and report once per interval."""
        del pl_module, outputs, batch, batch_idx
        self._last_batch_end = self._clock()
        self._compute_s += self._last_batch_end - self._batch_start
        self._steps += 1

        if self._window_start is None:
            self._window_start = self._batch_start

        elapsed_s = self._last_batch_end - self._window_start

        if elapsed_s < self._interval_s:
            return

        data_wait = self._data_wait_s / max(self._data_wait_s + self._compute_s, 1e-9)
        self._emit(
            f"[train] {self._steps / elapsed_s:.1f} it/s | data wait {data_wait:.0%} "
            f"| epoch {trainer.current_epoch} step {trainer.global_step}"
        )
        self._window_start = self._last_batch_end
        self._data_wait_s = self._compute_s = 0.0
        self._steps = 0


class CheckpointUploadCallback(pl.Callback):
    """Upload the checkpoints of a run at each epoch start, at fit end and when fit fails.

    Attributes:
        checkpoint_store: Where the checkpoints are uploaded.
        run_id: Run the checkpoints belong to.
        checkpoint_dir: Local folder the ``ModelCheckpoint`` callback writes to.
    """

    def __init__(
        self, checkpoint_store: MLFlowCheckpointStore, run_id: str, checkpoint_dir: Path
    ) -> None:
        """Store the destination and the folder to upload.

        Args:
            checkpoint_store: Where the checkpoints are uploaded.
            run_id: Run the checkpoints belong to.
            checkpoint_dir: Local folder the ``ModelCheckpoint`` callback writes to.
        """
        self.checkpoint_store = checkpoint_store
        self.run_id = run_id
        self.checkpoint_dir = checkpoint_dir

    @override
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Upload the checkpoints saved at the end of the previous epoch."""
        del trainer, pl_module
        # ModelCheckpoint always runs after user callbacks, so at epoch end it has not saved yet.
        self._upload()

    @override
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Upload the final checkpoints."""
        del trainer, pl_module
        self._upload()

    @override
    def on_exception(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException
    ) -> None:
        """Upload what was saved before the failure, so the run can be restarted from it."""
        del trainer, pl_module, exception
        self._upload()

    def _upload(self) -> None:
        """Upload ``checkpoint_dir`` if it holds a checkpoint."""
        if any(self.checkpoint_dir.glob("*.ckpt")):
            self.checkpoint_store.upload(self.run_id, self.checkpoint_dir)
