"""Tests for the training-loop callbacks."""

from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl

from baseline_core.core.ml.tracking.checkpoint import MLFlowCheckpointStore
from baseline_core.core.ml.training.callbacks import CheckpointUploadCallback, ThroughputMonitor


class TestThroughputMonitor:
    """Tests for ``ThroughputMonitor`` driven by a fake clock."""

    CLOCK_TICKS: ClassVar[list[float]] = [0, 1, 4, 5, 10]
    INTERVAL_S: ClassVar[float] = 10

    @pytest.fixture
    def trainer(self) -> MagicMock:
        """A trainer at step 2 of epoch 0."""
        trainer = MagicMock(spec=pl.Trainer)
        trainer.current_epoch = 0
        trainer.global_step = 2
        return trainer

    @pytest.fixture
    def lines(self) -> list[str]:
        """Lines the monitor emits."""
        return []

    @pytest.fixture
    def monitor(self, lines: list[str]) -> ThroughputMonitor:
        """A monitor reporting once the two batches fill its interval."""
        return ThroughputMonitor(
            self.INTERVAL_S, emit=lines.append, clock=iter(self.CLOCK_TICKS).__next__
        )

    @pytest.fixture
    def patient_monitor(self, lines: list[str]) -> ThroughputMonitor:
        """A monitor whose interval is longer than the two batches."""
        return ThroughputMonitor(
            self.INTERVAL_S + 1, emit=lines.append, clock=iter(self.CLOCK_TICKS).__next__
        )

    def run_two_batches(self, monitor: ThroughputMonitor, trainer: MagicMock) -> None:
        """Drive the hooks Lightning calls for one epoch start and two batches."""
        module = MagicMock(spec=pl.LightningModule)
        monitor.on_train_epoch_start(trainer, module)

        for batch_index in range(2):
            monitor.on_train_batch_start(trainer, module, None, batch_index)
            monitor.on_train_batch_end(trainer, module, None, None, batch_index)

    def test__on_train_batch_end__after_interval__should_emit_speed_and_data_wait(
        self, monitor: ThroughputMonitor, trainer: MagicMock, lines: list[str]
    ) -> None:
        """2 steps in 10 s, 2 of the 10 s spent waiting for batches."""
        self.run_two_batches(monitor, trainer)

        assert lines == ["[train] 0.2 it/s | data wait 20% | epoch 0 step 2"]

    def test__on_train_batch_end__before_interval__should_emit_nothing(
        self, patient_monitor: ThroughputMonitor, trainer: MagicMock, lines: list[str]
    ) -> None:
        """Nothing is reported until a full interval has passed."""
        self.run_two_batches(patient_monitor, trainer)

        assert lines == []


class TestCheckpointUploadCallback:
    """Tests for ``CheckpointUploadCallback``."""

    @pytest.fixture
    def checkpoint_store(self) -> MagicMock:
        """A store recording uploads."""
        return MagicMock(spec=MLFlowCheckpointStore)

    @pytest.fixture
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        """A folder where ``ModelCheckpoint`` already saved a checkpoint."""
        (tmp_path / "last.ckpt").write_text("last")
        return tmp_path

    @pytest.fixture
    def callback(
        self, checkpoint_store: MagicMock, checkpoint_dir: Path
    ) -> CheckpointUploadCallback:
        """Callback uploading ``checkpoint_dir`` for ``run-1``."""
        return CheckpointUploadCallback(checkpoint_store, "run-1", checkpoint_dir)

    def test__on_fit_end__should_upload_the_checkpoint_dir_of_the_run(
        self, callback: CheckpointUploadCallback, checkpoint_store: MagicMock, checkpoint_dir: Path
    ) -> None:
        """The whole checkpoint folder goes to the store, under the run id."""
        callback.on_fit_end(MagicMock(spec=pl.Trainer), MagicMock(spec=pl.LightningModule))

        checkpoint_store.upload.assert_called_once_with("run-1", checkpoint_dir)

    def test__on_train_epoch_start__should_upload_checkpoints_of_previous_epoch(
        self, callback: CheckpointUploadCallback, checkpoint_store: MagicMock, checkpoint_dir: Path
    ) -> None:
        """Checkpoints reach the store during training, so a crashed job can be restarted."""
        callback.on_train_epoch_start(
            MagicMock(spec=pl.Trainer), MagicMock(spec=pl.LightningModule)
        )

        checkpoint_store.upload.assert_called_once_with("run-1", checkpoint_dir)

    def test__on_exception__should_upload_checkpoints(
        self, callback: CheckpointUploadCallback, checkpoint_store: MagicMock, checkpoint_dir: Path
    ) -> None:
        """A failing run still leaves its latest checkpoints in the store."""
        callback.on_exception(
            MagicMock(spec=pl.Trainer), MagicMock(spec=pl.LightningModule), RuntimeError("oom")
        )

        checkpoint_store.upload.assert_called_once_with("run-1", checkpoint_dir)

    def test__on_train_epoch_start__when_no_checkpoint_yet__should_not_upload(
        self, checkpoint_store: MagicMock, tmp_path: Path
    ) -> None:
        """The first epoch has nothing to upload."""
        callback = CheckpointUploadCallback(checkpoint_store, "run-1", tmp_path / "empty")

        callback.on_train_epoch_start(
            MagicMock(spec=pl.Trainer), MagicMock(spec=pl.LightningModule)
        )

        checkpoint_store.upload.assert_not_called()
