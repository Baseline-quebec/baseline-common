"""Bridge the kit onto MLflow -- the only module in the kit that imports the library."""

import json
import subprocess  # noqa: S404 -- reads the git revision, fixed argv only
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any, ClassVar

import mlflow
import mlflow.pydantic_ai
from mlflow import MlflowClient
from mlflow.entities import Dataset, DatasetInput, Param
from mlflow.entities import Metric as MlflowMetric
from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.evalkit.case import EvalDataset
from baseline_core.core.llm.evalkit.eval_run import EvalRun
from baseline_core.core.llm.evalkit.evaluator import EvalResult, Evaluator


class MlflowTrackerConfig(BaseModel):
    """Where a tracker files its runs, and what it tags them with."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    experiment: str
    """The experiment every run of the suite is filed under."""

    store: Path
    """The folder holding the tracking database and the run artifacts."""

    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    """Run metadata to compare runs by, such as the model name."""

    tags: list[str] = Field(default_factory=list)
    """Labels to filter runs by across experiments, such as the service layer."""

    database: str = "mlflow.db"
    """The SQLite file under the store holding experiments, runs, params and metrics."""

    artifacts: str = "artifacts"
    """The folder under the store holding each run's files."""

    run_file: str = "run.json"
    """The artifact carrying the whole run, per-case scores and reasons included."""

    @property
    def tracking_uri(self) -> str:
        """The store's SQLite database, as MLflow addresses it."""
        return f"sqlite:///{(self.store / self.database).resolve()}"


class MlflowTracker[I, E, O]:
    """Evaluates a dataset as one MLflow run: every agent call traced, every score logged."""

    EXPERIMENT_KIND: ClassVar[dict[str, str]] = {
        "mlflow.experimentKind": "custom_model_development"
    }
    """The experiment tag that makes MLflow's UI show the runs table."""

    def __init__(self, evaluator: Evaluator[I, E, O], config: MlflowTrackerConfig) -> None:
        """Bind the evaluator to track and where its runs are filed.

        Args:
            evaluator: The evaluator scoring the system under test.
            config: The experiment, the store, and what each run is tagged with.
        """
        self._evaluator = evaluator
        self._config = config

    def evaluate(self, dataset: EvalDataset[I, E]) -> list[EvalResult[I, E, O]]:
        """Evaluate a dataset inside one MLflow run, tracing every agent call and its scores.

        Args:
            dataset: The named collection of labeled cases.

        Returns:
            The evaluator's results, unchanged.
        """
        client = self._client()
        experiment_id = self._experiment_id(client)
        previous = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(self._config.tracking_uri)
        mlflow.pydantic_ai.autolog(silent=True)

        try:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=dataset.name,
                tags={**dict.fromkeys(self._config.tags, "true"), **self._revision()},
            ) as active:
                results = self._evaluator.evaluate(dataset)
                self._log(client, active.info.run_id, dataset, results)
        finally:
            mlflow.flush_trace_async_logging()
            mlflow.pydantic_ai.autolog(disable=True, silent=True)
            mlflow.set_tracking_uri(previous)

        return results

    def _log(
        self,
        client: MlflowClient,
        run_id: str,
        dataset: EvalDataset[I, E],
        results: list[EvalResult[I, E, O]],
    ) -> None:
        """Record the params, each metric's mean and pass rate, the cost, the dataset, the cases.

        Args:
            client: The client bound to the store.
            run_id: The active run.
            dataset: The dataset the run was launched on.
            results: The per-case results that came back.
        """
        run: EvalRun[I, E, O, None] = EvalRun.build(dataset, results)
        stamp = int(run.ran_at.timestamp() * 1000)
        params = {"dataset": dataset.name, **self._config.hyperparameters}
        client.log_batch(
            run_id,
            metrics=[
                MlflowMetric(key=key, value=value, timestamp=stamp, step=0)
                for key, value in self._values(dataset, run).items()
            ],
            params=[Param(key=key, value=str(value)) for key, value in params.items()],
        )
        client.log_inputs(run_id, datasets=[self._input(dataset)])
        client.log_text(run_id, run.model_dump_json(indent=2), self._config.run_file)

    @staticmethod
    def _input(dataset: EvalDataset[I, E]) -> DatasetInput:
        """Describe the dataset a run scored, fingerprinted by its cases.

        Args:
            dataset: The dataset the run was launched on.

        Returns:
            The input MLflow lists beside the run; its digest changes whenever a case does.
        """
        digest = sha256(dataset.model_dump_json().encode()).hexdigest()[:8]
        return DatasetInput(
            Dataset(
                name=dataset.name,
                digest=digest,
                source_type="code",
                source=json.dumps({"dataset": dataset.name}),
                profile=json.dumps({"num_rows": len(dataset.cases)}),
            )
        )

    def _values(self, dataset: EvalDataset[I, E], run: EvalRun[I, E, O, None]) -> dict[str, float]:
        """Fold a run into the numbers followed from one run to the next.

        Args:
            dataset: The dataset the run was launched on.
            run: The run, stamped and costed.

        Returns:
            Each metric's mean and pass rate, the case counts, and whichever cost was reported.
        """
        values = {
            "cases": float(len(run.results)),
            "cases_dropped": float(len(dataset.cases) - len(run.results)),
        }

        for metric in self._evaluator.metrics:
            scored = [result.scores[type(metric).__name__] for result in run.results]

            if not scored:
                continue

            values[f"{metric.label}/mean"] = fmean(score.score for score in scored)
            values[f"{metric.label}/pass_rate"] = fmean(
                float(metric.passes(score)) for score in scored
            )

        cost = {
            "cost/input_tokens": run.cost.input_tokens,
            "cost/output_tokens": run.cost.output_tokens,
            "cost/elapsed_s": run.cost.elapsed_s,
        }
        values.update({key: float(value) for key, value in cost.items() if value is not None})
        return values

    @staticmethod
    def _revision() -> dict[str, str]:
        """Read the Git revision of the checkout the suite runs from, as tags the UI filters.

        Returns:
            The commit, the branch and whether the tree was dirty; empty when Git is unreadable.
        """
        # The working directory, not this file: the kit may be installed outside the checkout.
        try:
            read = [
                subprocess.run(  # noqa: S603 -- fixed argv, no shell, no outside input
                    ["git", *args],  # noqa: S607 -- whichever git the suite's PATH finds
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
                for args in (["rev-parse", "HEAD"], ["branch", "--show-current"], ["status", "-s"])
            ]
        except (OSError, subprocess.CalledProcessError):
            return {}

        commit, branch, status = read
        return {"git.commit": commit, "git.branch": branch, "git.dirty": str(bool(status)).lower()}

    def _client(self) -> MlflowClient:
        """Open a client on the store's SQLite database, creating the store when missing.

        Returns:
            A client bound to this bridge's store.
        """
        self._config.store.mkdir(parents=True, exist_ok=True)
        return MlflowClient(tracking_uri=self._config.tracking_uri)

    def _experiment_id(self, client: MlflowClient) -> str:
        """Find this suite's experiment, creating it with its artifacts under the store.

        Args:
            client: The client bound to the store.

        Returns:
            The experiment id, tagged with its kind and the suite's labels.
        """
        found = client.get_experiment_by_name(self._config.experiment)

        if found is None:
            artifacts = (self._config.store / self._config.artifacts).resolve().as_uri()
            experiment_id = client.create_experiment(
                self._config.experiment, artifact_location=artifacts
            )
        else:
            experiment_id = str(found.experiment_id)

        tags = {**self.EXPERIMENT_KIND, **dict.fromkeys(self._config.tags, "true")}

        for key, value in tags.items():
            client.set_experiment_tag(experiment_id, key, value)

        return experiment_id
