"""MLflow run management."""

import os
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Self
from urllib.parse import urlparse

import mlflow
from loguru import logger
from pydantic import BaseModel

from baseline_core.core.ml.tracking.system_context import SystemContext


if TYPE_CHECKING:
    from mlflow.entities import Run


class TrackingConfig(BaseModel):
    """MLflow server, experiment, run name and the artifacts a new run logs.

    Attributes:
        experiment_name: MLflow experiment the runs are logged under.
        run_name: MLflow run name, or ``None`` to let MLflow pick one.
        tracking_uri: MLflow tracking URI, or ``None`` to use ``resolved_tracking_uri``.
        requirements_artifact: Artifact holding the frozen dependencies of a new run.
    """

    DEFAULT_TRACKING_URI: ClassVar[str] = "sqlite:///experiments/mlflow.db"

    experiment_name: str
    run_name: str | None = None
    tracking_uri: str | None = None
    requirements_artifact: str = "requirements_frozen.txt"

    @property
    def resolved_tracking_uri(self) -> str:
        """``tracking_uri``, else ``MLFLOW_TRACKING_URI``, else the local SQLite store."""
        return (
            self.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or self.DEFAULT_TRACKING_URI
        )


class MLFlowTracker:
    """One MLflow run, usable as a context manager.

    Attributes:
        config: Experiment, run name, tracking store and the artifacts a new run logs.
        nested: Whether the run is nested under the currently active one.
        experiment_id: Resolved experiment id.
        active_run: The run while it is open, else ``None``.
    """

    def __init__(self, config: TrackingConfig, *, nested: bool = False) -> None:
        """Select the experiment, creating it if needed. No run is started yet.

        Args:
            config: Experiment, run name and tracking store, resolved by ``resolved_tracking_uri``.
            nested: Whether the run is nested under the currently active one.
        """
        self.config = config
        self.nested = nested
        mlflow.set_tracking_uri(config.resolved_tracking_uri)
        self.experiment_id: str = mlflow.set_experiment(config.experiment_name).experiment_id
        self.active_run: Run | None = None

    @property
    def artifact_path(self) -> Path:
        """Local artifact directory of the active run, where checkpoints can be written."""
        artifact_uri = self._require_run().info.artifact_uri

        if artifact_uri is None:
            error_msg = "The active run has no artifact location."
            raise RuntimeError(error_msg)

        return self.local_artifact_dir(artifact_uri)

    @staticmethod
    def local_artifact_dir(artifact_uri: str) -> Path:
        """Local directory behind a ``file://`` (or bare path) artifact URI.

        Args:
            artifact_uri: MLflow artifact URI of a run.

        Returns:
            The resolved local path.

        Raises:
            ValueError: If the artifact store is remote.
        """
        parsed = urlparse(artifact_uri)

        if parsed.scheme not in {"file", ""}:
            error_msg = f"Artifact store '{artifact_uri}' is not local."
            raise ValueError(error_msg)

        return Path(parsed.path).resolve()

    @staticmethod
    def flatten_dict(
        data: Mapping[str, Any], parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Flatten nested mappings into dotted keys: ``{"a": {"b": 1}}`` becomes ``{"a.b": 1}``.

        Args:
            data: Possibly nested mapping.
            parent_key: Prefix of every key, used by the recursion.
            sep: Separator between nesting levels.

        Returns:
            A flat mapping.
        """
        items: dict[str, Any] = {}

        for key, value in data.items():
            full_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, Mapping):
                items.update(MLFlowTracker.flatten_dict(value, full_key, sep))
            else:
                items[full_key] = value

        return items

    def __enter__(self) -> Self:
        """Start the run."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """End the run."""
        self.end()

    def start(self, run_id: str | None = None) -> None:
        """Start a new run, or resume ``run_id``.

        Args:
            run_id: Run to resume, or ``None`` for a new run logging its system context.
        """
        self.active_run = mlflow.start_run(
            run_id=run_id,
            experiment_id=self.experiment_id,
            run_name=self.config.run_name,
            nested=self.nested,
        )
        action = "Resumed" if run_id else "Started"
        logger.info("{} MLflow run {}", action, self.active_run.info.run_id)

        if run_id is None:
            self.log_params(SystemContext.collect())
            mlflow.log_text(SystemContext.freeze_dependencies(), self.config.requirements_artifact)

    def end(self) -> None:
        """End the active run, if any."""
        if self.active_run is None:
            return

        logger.info("Ending MLflow run {}", self.active_run.info.run_id)
        mlflow.end_run()
        self.active_run = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters on the active run."""
        self._require_run()
        mlflow.log_params(dict(params))

    def log_config(self, config: BaseModel | Mapping[str, Any]) -> None:
        """Log a config as parameters, flattened to dotted keys (``data.batch_size``)."""
        data = config.model_dump() if isinstance(config, BaseModel) else config
        self.log_params(self.flatten_dict(data))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log scalar metrics on the active run, optionally at a given step."""
        self._require_run()
        mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, local_path: Path | str, artifact_path: str | None = None) -> None:
        """Upload a local file or directory to the active run.

        Args:
            local_path: File or directory to upload.
            artifact_path: Destination directory within the run's artifacts.

        Raises:
            FileNotFoundError: If ``local_path`` does not exist.
        """
        self._require_run()

        if not Path(local_path).exists():
            error_msg = f"Artifact not found: {local_path}"
            raise FileNotFoundError(error_msg)

        mlflow.log_artifact(str(local_path), artifact_path)

    def _require_run(self) -> "Run":
        """Return the active run; logging without one is a programming error, so fail fast."""
        if self.active_run is None:
            error_msg = "No active run. Call start() or use the tracker as a context manager."
            raise RuntimeError(error_msg)

        return self.active_run
