"""Tests for the MLflow tracker and its helpers."""

from pathlib import Path
from typing import ClassVar

import mlflow
import pytest
from pydantic import BaseModel

from baseline_core.core.ml.tracking.tracker import MLFlowTracker, TrackingConfig


class TestTrackingConfig:
    """Tests for ``TrackingConfig.resolved_tracking_uri``."""

    def test__resolved_tracking_uri__when_set__should_return_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit URI wins over the environment."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env")
        config = TrackingConfig(experiment_name="e", tracking_uri="http://explicit")

        assert config.resolved_tracking_uri == "http://explicit"

    def test__resolved_tracking_uri__when_unset__should_read_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Azure ML sets ``MLFLOW_TRACKING_URI`` in jobs."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env")

        assert TrackingConfig(experiment_name="e").resolved_tracking_uri == "http://from-env"

    def test__resolved_tracking_uri__when_nothing_set__should_use_local_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without configuration runs go to the local SQLite store."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        config = TrackingConfig(experiment_name="e")
        assert config.resolved_tracking_uri == TrackingConfig.DEFAULT_TRACKING_URI


class TestFlattenDict:
    """Tests for ``MLFlowTracker.flatten_dict()``."""

    def test__flatten_dict__when_nested__should_join_keys_with_dots(self) -> None:
        """Nested keys become dotted paths."""
        data = {"data": {"batch_size": 32, "loader": {"workers": 4}}, "lr": 0.1}

        flat = MLFlowTracker.flatten_dict(data)

        assert flat == {"data.batch_size": 32, "data.loader.workers": 4, "lr": 0.1}

    def test__flatten_dict__when_custom_separator__should_use_it(self) -> None:
        """The separator is configurable."""
        flat = MLFlowTracker.flatten_dict({"a": {"b": 1}}, sep="/")

        assert flat == {"a/b": 1}


class TestLocalArtifactDir:
    """Tests for ``MLFlowTracker.local_artifact_dir()``."""

    def test__local_artifact_dir__when_file_uri__should_return_path(self, tmp_path: Path) -> None:
        """``file://`` URIs resolve to their path."""
        path = MLFlowTracker.local_artifact_dir(tmp_path.as_uri())

        assert path == tmp_path.resolve()

    def test__local_artifact_dir__when_bare_path__should_return_path(self, tmp_path: Path) -> None:
        """A path without scheme is local too."""
        assert MLFlowTracker.local_artifact_dir(str(tmp_path)) == tmp_path.resolve()

    def test__local_artifact_dir__when_remote__should_raise_value_error(self) -> None:
        """Remote stores have no local directory."""
        with pytest.raises(ValueError, match="is not local"):
            MLFlowTracker.local_artifact_dir("s3://bucket/artifacts")


class Config(BaseModel):
    """Nested config to log."""

    lr: float = 0.1
    data: dict[str, int] = {"batch_size": 32}


class TestMLFlowTracker:
    """Tests for ``MLFlowTracker`` against a temporary file store."""

    EXPERIMENT: ClassVar[str] = "tracker_tests"

    @pytest.fixture
    def tracker(self, tracking_uri: str) -> MLFlowTracker:
        """Tracker on the temporary store, run not started."""
        return MLFlowTracker(
            TrackingConfig(
                experiment_name=self.EXPERIMENT, run_name="run", tracking_uri=tracking_uri
            )
        )

    def test__init__when_uri_unset__should_use_resolved_tracking_uri(
        self, tracking_uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an explicit URI the tracker follows ``MLFLOW_TRACKING_URI``, as in Azure ML."""
        env_uri = f"sqlite:///{tmp_path / 'from_env.db'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", env_uri)

        MLFlowTracker(TrackingConfig(experiment_name=self.EXPERIMENT))

        assert tracking_uri != env_uri
        assert mlflow.get_tracking_uri() == env_uri

    def test__log_params__when_no_active_run__should_raise_runtime_error(
        self, tracker: MLFlowTracker
    ) -> None:
        """Logging outside a run is a programming error, not a silent MLflow default."""
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_params({"lr": 0.1})

    def test__context_manager__should_start_and_end_run(self, tracker: MLFlowTracker) -> None:
        """The run is active inside the block and closed after."""
        with tracker:
            run_id = tracker.active_run.info.run_id if tracker.active_run else None

        assert run_id is not None
        assert tracker.active_run is None
        assert mlflow.MlflowClient().get_run(run_id).info.status == "FINISHED"

    def test__start__when_new_run__should_log_system_context_and_dependencies(
        self, tracker: MLFlowTracker
    ) -> None:
        """Hostname, git commit and the frozen dependencies are recorded."""
        with tracker:
            run_id = tracker.active_run.info.run_id if tracker.active_run else ""

        client = mlflow.MlflowClient()
        params = client.get_run(run_id).data.params
        assert "system.hostname" in params
        assert "git.commit" in params
        artifacts = [artifact.path for artifact in client.list_artifacts(run_id)]
        assert tracker.config.requirements_artifact in artifacts

    def test__log_config__should_flatten_nested_fields(self, tracker: MLFlowTracker) -> None:
        """Nested config fields are logged as dotted parameters."""
        with tracker:
            tracker.log_config(Config())
            run_id = tracker.active_run.info.run_id if tracker.active_run else ""

        params = mlflow.MlflowClient().get_run(run_id).data.params
        assert params["lr"] == "0.1"
        assert params["data.batch_size"] == "32"

    def test__artifact_path__should_be_local_directory_of_run(
        self, tracker: MLFlowTracker, tmp_path: Path
    ) -> None:
        """Checkpoints can be written next to the run's artifacts."""
        with tracker:
            artifact_path = tracker.artifact_path

        assert artifact_path.is_dir()
        assert artifact_path.is_relative_to(tmp_path)

    def test__log_artifact__when_file_missing__should_raise_file_not_found(
        self, tracker: MLFlowTracker, tmp_path: Path
    ) -> None:
        """A missing artifact is reported before calling MLflow."""
        with tracker, pytest.raises(FileNotFoundError, match="Artifact not found"):
            tracker.log_artifact(tmp_path / "missing.onnx")
