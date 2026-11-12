"""Checkpoint store: where runs upload their checkpoints and later runs load them from."""

from pathlib import Path
from typing import ClassVar, Literal, Self

import mlflow
from loguru import logger
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, model_validator

from baseline_core.core.ml.data.phase import Phase


class CheckpointResolution(BaseModel):
    """Checkpoint to load and the run it comes from.

    Attributes:
        checkpoint_path: Local checkpoint to load, or ``None`` when there is none.
        run_id: Run that wrote the checkpoint, or ``None`` when there is none.
    """

    checkpoint_path: Path | None = None
    run_id: str | None = None


class CheckpointConfig(BaseModel):
    """Where the checkpoint comes from and what is done with it.

    Attributes:
        run_id: Source run (on Azure ML, the previous job's name), or ``None`` for the latest.
        which_checkpoint: ``"best"`` validation score or ``"last"`` finished epoch of the run.
        mode: Phase the checkpoint is loaded for; evaluation phases require ``"best"``.
    """

    run_id: str | None = None
    which_checkpoint: Literal["best", "last"] = "best"
    mode: Phase = Phase.TRAIN

    @model_validator(mode="after")
    def validate_evaluation_uses_best(self) -> Self:
        """Evaluating the last checkpoint rather than the best one is almost always a mistake."""
        if self.is_evaluation and self.which_checkpoint == "last":
            error_msg = "Evaluation must use the best checkpoint; set which_checkpoint='best'."
            raise ValueError(error_msg)

        return self

    @property
    def is_evaluation(self) -> bool:
        """Whether the checkpoint is loaded to evaluate rather than to train."""
        return self.mode in {Phase.VAL, Phase.TEST}


class MLFlowCheckpointStore:
    """Checkpoints kept as MLflow run artifacts under ``checkpoints/``.

    Attributes:
        client: MLflow client of the tracking server.
        cache_dir: Local folder receiving downloaded checkpoints, one subfolder per run.
    """

    ARTIFACT_DIR: ClassVar[str] = "checkpoints"

    def __init__(self, tracking_uri: str, cache_dir: Path) -> None:
        """Connect to the tracking server.

        Args:
            tracking_uri: MLflow tracking URI.
            cache_dir: Local folder receiving downloaded checkpoints.
        """
        self.client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        self.cache_dir = cache_dir

    def upload(self, run_id: str, checkpoint_dir: Path) -> None:
        """Log every file of ``checkpoint_dir`` as an artifact of ``run_id``."""
        self.client.log_artifacts(run_id, str(checkpoint_dir), self.ARTIFACT_DIR)
        logger.info("Uploaded {} to run {}", checkpoint_dir, run_id)

    def latest(
        self, experiment_name: str, which_checkpoint: Literal["best", "last"]
    ) -> CheckpointResolution:
        """Download ``checkpoints/<which_checkpoint>.ckpt`` of the latest run of the experiment."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        runs = (
            self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=1,
            )
            if experiment is not None
            else []
        )

        if not runs:
            logger.info("No previous run in '{}'.", experiment_name)
            return CheckpointResolution()

        run_id = runs[0].info.run_id
        checkpoint_path = self._download(run_id, which_checkpoint)

        if checkpoint_path is None:
            logger.warning("Run {} has no {} checkpoint.", run_id, which_checkpoint)
            return CheckpointResolution()

        return CheckpointResolution(checkpoint_path=checkpoint_path, run_id=run_id)

    def from_run(
        self, run_id: str, which_checkpoint: Literal["best", "last"]
    ) -> CheckpointResolution:
        """Download ``checkpoints/<which_checkpoint>.ckpt`` of run ``run_id``."""
        try:
            self.client.get_run(run_id)
        except MlflowException as error:
            error_msg = f"Run '{run_id}' not found on the tracking server."
            raise ValueError(error_msg) from error

        checkpoint_path = self._download(run_id, which_checkpoint)

        if checkpoint_path is None:
            error_msg = f"Run '{run_id}' has no {which_checkpoint} checkpoint."
            raise ValueError(error_msg)

        return CheckpointResolution(checkpoint_path=checkpoint_path, run_id=run_id)

    def _download(self, run_id: str, which_checkpoint: Literal["best", "last"]) -> Path | None:
        """Download a checkpoint of ``run_id`` to the cache, or ``None`` if it was not logged."""
        artifact = f"{self.ARTIFACT_DIR}/{which_checkpoint}.ckpt"
        logged_paths = {
            item.path for item in self.client.list_artifacts(run_id, self.ARTIFACT_DIR)
        }

        if artifact not in logged_paths:
            return None

        destination = self.cache_dir / run_id
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(self.client.download_artifacts(run_id, artifact, str(destination)))
        logger.info("Downloaded {} of run {}", artifact, run_id)
        return checkpoint_path

    def resolve(self, load_config: CheckpointConfig, experiment_name: str) -> CheckpointResolution:
        """Download the checkpoint ``load_config`` asks for.

        Args:
            load_config: The run (or the latest of the experiment) and checkpoint to load.
            experiment_name: Experiment searched when ``load_config.run_id`` is ``None``.

        Returns:
            The local checkpoint and its run; empty when the experiment has none.
        """
        if load_config.run_id is not None:
            return self.from_run(load_config.run_id, load_config.which_checkpoint)

        return self.latest(experiment_name, load_config.which_checkpoint)
