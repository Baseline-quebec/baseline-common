"""Tests for checkpoint load settings and the MLflow checkpoint store."""

from pathlib import Path
from typing import ClassVar, Literal

import mlflow
import pytest

from baseline_core.core.ml.data.phase import Phase
from baseline_core.core.ml.tracking.checkpoint import (
    CheckpointConfig,
    CheckpointResolution,
    MLFlowCheckpointStore,
)


class TestCheckpointLoadConfig:
    """Tests for the validation of ``CheckpointLoadConfig``."""

    @pytest.mark.parametrize("mode", [Phase.VAL, Phase.TEST])
    def test__init__when_evaluating_last_checkpoint__should_raise_value_error(
        self, mode: Phase
    ) -> None:
        """Evaluation must use the best checkpoint."""
        with pytest.raises(ValueError, match="Evaluation must use the best checkpoint"):
            CheckpointConfig(which_checkpoint="last", mode=mode)

    def test__init__when_training_from_last_checkpoint__should_be_allowed(self) -> None:
        """Training may continue from the last checkpoint."""
        load_config = CheckpointConfig(which_checkpoint="last", mode=Phase.TRAIN)

        assert load_config.is_evaluation is False


class TestMLFlowCheckpointStore:
    """Tests for ``MLFlowCheckpointStore`` against a temporary MLflow store."""

    EXPERIMENT_NAME: ClassVar[str] = "checkpoint_tests"

    @pytest.fixture
    def checkpoint_store(self, tracking_uri: str, tmp_path: Path) -> MLFlowCheckpointStore:
        """Store on the temporary server, caching downloads under ``tmp_path``."""
        return MLFlowCheckpointStore(tracking_uri, tmp_path / "cache")

    @pytest.fixture
    def run_id(self, tracking_uri: str) -> str:
        """One finished run in the experiment, without artifacts."""
        del tracking_uri
        mlflow.set_experiment(self.EXPERIMENT_NAME)

        with mlflow.start_run() as run:
            return str(run.info.run_id)

    @pytest.fixture
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        """Local folder with a ``best.ckpt`` and a ``last.ckpt``."""
        checkpoint_dir = tmp_path / "local_checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "best.ckpt").write_text("best")
        (checkpoint_dir / "last.ckpt").write_text("last")
        return checkpoint_dir

    def test__latest__when_no_experiment__should_return_empty_resolution(
        self, checkpoint_store: MLFlowCheckpointStore
    ) -> None:
        """An unknown experiment has no checkpoint."""
        resolution = checkpoint_store.latest(self.EXPERIMENT_NAME, "best")

        assert resolution == CheckpointResolution()

    @pytest.mark.usefixtures("run_id")
    def test__latest__when_run_without_checkpoint__should_return_empty_resolution(
        self, checkpoint_store: MLFlowCheckpointStore
    ) -> None:
        """A run that uploaded nothing has no checkpoint."""
        resolution = checkpoint_store.latest(self.EXPERIMENT_NAME, "best")

        assert resolution == CheckpointResolution()

    @pytest.mark.parametrize("which_checkpoint", ["best", "last"])
    def test__latest__when_checkpoints_uploaded__should_download_the_requested_one(
        self,
        checkpoint_store: MLFlowCheckpointStore,
        run_id: str,
        checkpoint_dir: Path,
        which_checkpoint: Literal["best", "last"],
    ) -> None:
        """Uploaded checkpoints come back from the cache, with the run that wrote them."""
        checkpoint_store.upload(run_id, checkpoint_dir)

        resolution = checkpoint_store.latest(self.EXPERIMENT_NAME, which_checkpoint)

        assert resolution.run_id == run_id
        assert resolution.checkpoint_path is not None
        assert resolution.checkpoint_path.read_text() == which_checkpoint
        assert resolution.checkpoint_path.is_relative_to(checkpoint_store.cache_dir)

    def test__from_run__should_download_that_run_even_if_not_latest(
        self, checkpoint_store: MLFlowCheckpointStore, run_id: str, checkpoint_dir: Path
    ) -> None:
        """A job id (the MLflow run id on Azure ML) picks its own checkpoint, not the latest."""
        checkpoint_store.upload(run_id, checkpoint_dir)

        with mlflow.start_run():
            pass

        resolution = checkpoint_store.from_run(run_id, "last")

        assert resolution.run_id == run_id
        assert resolution.checkpoint_path is not None
        assert resolution.checkpoint_path.read_text() == "last"

    def test__from_run__when_run_has_no_checkpoint__should_raise_value_error(
        self, checkpoint_store: MLFlowCheckpointStore, run_id: str
    ) -> None:
        """Asking for a job's checkpoint that does not exist must not silently start over."""
        with pytest.raises(ValueError, match=run_id):
            checkpoint_store.from_run(run_id, "best")

    @pytest.mark.usefixtures("tracking_uri")
    def test__from_run__when_run_unknown__should_raise_value_error(
        self, checkpoint_store: MLFlowCheckpointStore
    ) -> None:
        """A mistyped job id fails with the id in the message."""
        with pytest.raises(ValueError, match="no-such-run"):
            checkpoint_store.from_run("no-such-run", "best")

    def test__resolve__when_run_id_set__should_load_from_that_run(
        self, checkpoint_store: MLFlowCheckpointStore, run_id: str, checkpoint_dir: Path
    ) -> None:
        """``CheckpointLoadConfig.run_id`` selects the job to restart from."""
        checkpoint_store.upload(run_id, checkpoint_dir)

        with mlflow.start_run():
            pass

        load_config = CheckpointConfig(run_id=run_id, which_checkpoint="last")

        resolution = checkpoint_store.resolve(load_config, self.EXPERIMENT_NAME)

        assert resolution.run_id == run_id

    def test__resolve__when_run_id_unset__should_load_latest_run(
        self, checkpoint_store: MLFlowCheckpointStore, run_id: str, checkpoint_dir: Path
    ) -> None:
        """Without a run id the latest run of the experiment is used, as before."""
        checkpoint_store.upload(run_id, checkpoint_dir)

        resolution = checkpoint_store.resolve(CheckpointConfig(), self.EXPERIMENT_NAME)

        assert resolution.run_id == run_id
