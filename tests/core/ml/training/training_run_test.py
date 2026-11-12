"""Tests for ``TrainingRun`` on the dummy model, on CPU."""

from pathlib import Path

import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from baseline_core.core.ml.training.backbone_heads_module import BackboneHeadsModule
from baseline_core.core.ml.training.callbacks import ThroughputMonitor
from baseline_core.core.ml.training.training_run import TrainerConfig, TrainingRun
from tests.core.ml.dummy_classes import DummyBackbone
from tests.core.ml.training.dummy_classes import GroupedSampleDataModule


@pytest.mark.filterwarnings("ignore:The '.*_dataloader' does not have many workers")
class TestTrainingRun:
    """Tests for ``TrainingRun.fit``, ``test`` and ``load_weights``."""

    @pytest.fixture
    def trainer_config(self) -> TrainerConfig:
        """One epoch on CPU, logging every step."""
        return TrainerConfig(max_epochs=1, accelerator="cpu", log_every_n_steps=1)

    @pytest.fixture
    def checkpoint_callback(self, tmp_path: Path) -> ModelCheckpoint:
        """Best and last checkpoints written under ``tmp_path``."""
        return ModelCheckpoint(
            dirpath=tmp_path / "checkpoints", filename="best", monitor="val_loss", save_last=True
        )

    @pytest.fixture
    def training_run(self, trainer_config: TrainerConfig) -> TrainingRun:
        """A run without logger nor callbacks."""
        return TrainingRun(trainer_config)

    @pytest.fixture
    def checkpointed_run(
        self, trainer_config: TrainerConfig, checkpoint_callback: ModelCheckpoint
    ) -> TrainingRun:
        """A run saving its checkpoints through ``checkpoint_callback``."""
        return TrainingRun(trainer_config, callbacks=[checkpoint_callback])

    @pytest.fixture
    def unmonitored_run(self) -> TrainingRun:
        """A run with the throughput lines turned off."""
        return TrainingRun(TrainerConfig(max_epochs=1, accelerator="cpu", monitor_interval_s=None))

    def test__fit__when_checkpoint_callback__should_return_best_checkpoint(
        self,
        checkpointed_run: TrainingRun,
        bare_module: BackboneHeadsModule,
        grouped_datamodule: GroupedSampleDataModule,
    ) -> None:
        """The best checkpoint written by the injected callback is returned."""
        best_checkpoint = checkpointed_run.fit(bare_module, grouped_datamodule)

        assert best_checkpoint is not None
        assert best_checkpoint.name == "best.ckpt"
        assert best_checkpoint.exists()

    def test__fit__should_monitor_throughput_by_default(self, training_run: TrainingRun) -> None:
        """Training speed and data wait are reported unless turned off."""
        trainer = training_run._build_trainer()

        # Lightning sets Trainer.callbacks in its callback connector, without an annotation.
        callbacks = trainer.callbacks  # type: ignore[attr-defined]
        assert any(isinstance(callback, ThroughputMonitor) for callback in callbacks)

    def test__fit__when_monitor_disabled__should_not_monitor_throughput(
        self, unmonitored_run: TrainingRun
    ) -> None:
        """``monitor_interval_s=None`` turns the throughput lines off."""
        trainer = unmonitored_run._build_trainer()

        # Lightning sets Trainer.callbacks in its callback connector, without an annotation.
        callbacks = trainer.callbacks  # type: ignore[attr-defined]
        assert not any(isinstance(callback, ThroughputMonitor) for callback in callbacks)

    def test__fit__when_no_checkpoint_callback__should_return_none(
        self,
        training_run: TrainingRun,
        bare_module: BackboneHeadsModule,
        grouped_datamodule: GroupedSampleDataModule,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without a ``ModelCheckpoint`` nothing is saved."""
        monkeypatch.chdir(tmp_path)

        best_checkpoint = training_run.fit(bare_module, grouped_datamodule)

        assert best_checkpoint is None
        assert not any(tmp_path.rglob("*.ckpt"))

    def test__test__should_return_test_metrics(
        self,
        training_run: TrainingRun,
        bare_module: BackboneHeadsModule,
        grouped_datamodule: GroupedSampleDataModule,
    ) -> None:
        """The test loss logged by the module comes back as a float."""
        test_metrics = training_run.test(bare_module, grouped_datamodule)

        assert isinstance(test_metrics["test_loss"], float)

    def test__load_weights__should_restore_the_checkpoint_weights(
        self,
        checkpointed_run: TrainingRun,
        bare_module: BackboneHeadsModule,
        backbone: DummyBackbone,
        grouped_datamodule: GroupedSampleDataModule,
    ) -> None:
        """Weights changed after the checkpoint come back to their saved values."""
        best_checkpoint = checkpointed_run.fit(bare_module, grouped_datamodule)
        assert best_checkpoint is not None
        user_weight = backbone.get_parameter("layers.user.weight")
        saved_weights = user_weight.detach().clone()

        with torch.no_grad():
            user_weight.zero_()

        TrainingRun.load_weights(bare_module, best_checkpoint)

        assert torch.equal(backbone.get_parameter("layers.user.weight"), saved_weights)
