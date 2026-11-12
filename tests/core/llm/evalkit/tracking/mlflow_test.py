"""Unit tests for the MLflow tracker: what an evaluation leaves in a real SQLite store."""

import shutil
import subprocess  # noqa: S404 -- builds a throwaway Git checkout, fixed argv only
from collections import Counter
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.entities import Run
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from baseline_core.core.llm.evalkit.abstract_eval_adapter import AbstractEvalAdapter
from baseline_core.core.llm.evalkit.case import EvalCase, EvalDataset, EvalOutput
from baseline_core.core.llm.evalkit.eval_run import EvalRun
from baseline_core.core.llm.evalkit.evaluator import Evaluator
from baseline_core.core.llm.evalkit.tracking.mlflow import MlflowTracker, MlflowTrackerConfig
from tests.core.llm.evalkit.conftest import (
    FailingAdapter,
    InputCopyingAdapter,
    ItemRecallMetric,
    TokenBillingAdapter,
)


class AgentCallingAdapter(AbstractEvalAdapter[list[str], list[str]]):
    """Test double: echoes its input after one call to a pydantic-ai agent on its test model."""

    async def evaluate(self, case_input: list[str]) -> EvalOutput[list[str]]:
        """Call the agent once, then echo the input.

        Args:
            case_input: The case input.

        Returns:
            The echoed output.
        """
        await Agent(TestModel()).run(" ".join(case_input))
        return EvalOutput(value=case_input)


class TestEvaluate:
    """Evaluating a dataset as one MLflow run."""

    @pytest.fixture(scope="class")
    @classmethod
    def store(cls, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """A tracking store shared by the class's tests."""
        return tmp_path_factory.mktemp("mlflow")

    @pytest.fixture(scope="class")
    @classmethod
    def reader(cls, store: Path) -> MlflowClient:
        """A client reading back what the tracker wrote."""
        return MlflowClient(
            tracking_uri=MlflowTrackerConfig(experiment="", store=store).tracking_uri
        )

    @pytest.fixture
    def experiment(self, request: pytest.FixtureRequest) -> str:
        """An experiment of the test's own, named after it."""
        return str(request.node.name)

    @pytest.fixture
    def adapter(self) -> AbstractEvalAdapter[list[str], list[str]]:
        """The system under test: an echo, unless the test parametrizes another."""
        return InputCopyingAdapter()

    @pytest.fixture
    def config(self, store: Path, experiment: str) -> MlflowTrackerConfig:
        """The test's experiment in the shared store, with a model and labels as a suite has."""
        return MlflowTrackerConfig(
            experiment=experiment,
            store=store,
            hyperparameters={"model": "model-test"},
            tags=["ingestion", "matching"],
        )

    @pytest.fixture
    def tracker(
        self, config: MlflowTrackerConfig, adapter: AbstractEvalAdapter[list[str], list[str]]
    ) -> MlflowTracker[list[str], list[str], list[str]]:
        """A tracker over the overlap double."""
        return MlflowTracker(
            Evaluator(
                adapter, [ItemRecallMetric(ItemRecallMetric.PASS_MARK)], tolerate_errors=True
            ),
            config,
        )

    @pytest.fixture
    def experiment_id(self, config: MlflowTrackerConfig, reader: MlflowClient) -> str:
        """The test's experiment, created up front so the tracker finds it by name."""
        artifacts = (config.store / config.artifacts).resolve().as_uri()
        return reader.create_experiment(config.experiment, artifact_location=artifacts)

    @pytest.fixture
    def run(
        self,
        tracker: MlflowTracker[list[str], list[str], list[str]],
        reader: MlflowClient,
        experiment_id: str,
        dataset: EvalDataset[list[str], list[str]],
    ) -> Run:
        """The one run a tracked evaluation of the dataset leaves in the test's experiment."""
        tracker.evaluate(dataset)
        runs = reader.search_runs([experiment_id])
        assert len(runs) == 1
        return runs[0]

    def test_the_evaluator_results_come_back_unchanged(
        self,
        tracker: MlflowTracker[list[str], list[str], list[str]],
        dataset: EvalDataset[list[str], list[str]],
    ) -> None:
        """Tracking adds a record, never a change to what the gates are handed."""
        tracked = tracker.evaluate(dataset)
        bare = Evaluator(
            InputCopyingAdapter(), [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]
        ).evaluate(dataset)
        assert [result.scores for result in tracked] == [result.scores for result in bare]

    @pytest.mark.parametrize(
        "dataset",
        [
            EvalDataset[list[str], list[str]](
                name="mixed",
                cases=[
                    EvalCase(name="half", input=["a"], expected=["a", "b"]),
                    EvalCase(name="full", input=["c"], expected=["c"]),
                ],
            )
        ],
    )
    def test_each_metric_is_followed_by_its_mean_and_pass_rate(self, run: Run) -> None:
        """A half-matched case and a full one average to 0.75, and one of the two passes."""
        assert run.data.metrics["Item Recall/mean"] == pytest.approx(0.75)
        assert run.data.metrics["Item Recall/pass_rate"] == pytest.approx(0.5)

    @pytest.mark.parametrize("adapter", [FailingAdapter(only="c")])
    def test_a_dropped_case_is_counted_beside_the_means(self, run: Run) -> None:
        """A run short of one case says so, rather than reading as a clean one."""
        assert run.data.metrics["cases"] == 1
        assert run.data.metrics["cases_dropped"] == 1

    def test_the_dataset_and_the_hyperparameters_are_params(self, run: Run) -> None:
        """Runs are compared by the dataset they scored and the model they ran on."""
        assert run.data.params == {"dataset": "mini_corpus", "model": "model-test"}
        assert run.info.run_name == "mini_corpus"

    @pytest.mark.parametrize("adapter", [TokenBillingAdapter()])
    def test_a_billed_cost_is_logged(self, run: Run) -> None:
        """Token counts and wall time land when the cases billed them."""
        assert run.data.metrics["cost/input_tokens"] == 3
        assert run.data.metrics["cost/output_tokens"] == 6
        assert run.data.metrics["cost/elapsed_s"] == 1.0

    def test_an_unbilled_cost_is_left_out(self, run: Run) -> None:
        """A system that bills nothing leaves no cost metric at all."""
        assert not any(key.startswith("cost/") for key in run.data.metrics)

    def test_the_whole_run_is_kept_as_an_artifact(
        self, run: Run, reader: MlflowClient, config: MlflowTrackerConfig, tmp_path: Path
    ) -> None:
        """The per-case scores and reasons survive in the run file, beside the means."""
        path = reader.download_artifacts(run.info.run_id, config.run_file, str(tmp_path))
        text = Path(path).read_text(encoding="utf-8")
        kept: EvalRun[list[str], list[str], list[str], None] = EvalRun.model_validate_json(text)
        assert kept.dataset == "mini_corpus"
        assert [result.case.name for result in kept.results] == ["first", "second"]
        assert (config.store / config.artifacts).is_dir()

    @staticmethod
    def checkout(path: Path) -> str:
        """Make a Git checkout of one commit on ``main``, whatever the host repository is.

        Args:
            path: The folder to make the checkout in.

        Returns:
            The commit.
        """
        git = shutil.which("git")
        assert git is not None
        # The host's identity and signing settings must not decide whether the commit is made.
        isolated = [git, "-c", "user.name=Test", "-c", "user.email=test@example.com"]
        (path / "suite.py").write_text("SCORED = True\n", encoding="utf-8")

        for args in (
            ["init", "--quiet", "--initial-branch=main"],
            ["add", "suite.py"],
            ["-c", "commit.gpgsign=false", "commit", "--quiet", "--message", "Score the suite"],
        ):
            subprocess.run([*isolated, *args], cwd=path, check=True)  # noqa: S603 -- fixed argv

        return subprocess.run(  # noqa: S603 -- fixed argv
            [git, "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()

    def test_the_run_is_tagged_with_the_git_revision_it_ran_on(
        self,
        tracker: MlflowTracker[list[str], list[str], list[str]],
        reader: MlflowClient,
        experiment_id: str,
        dataset: EvalDataset[list[str], list[str]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The commit, its branch and the dirty flag name the checkout that scored the run."""
        commit = self.checkout(tmp_path)
        monkeypatch.chdir(tmp_path)

        tracker.evaluate(dataset)
        (run,) = reader.search_runs([experiment_id])

        assert len(commit) == 40
        assert run.data.tags["git.commit"] == commit
        assert run.data.tags["git.branch"] == "main"
        assert run.data.tags["git.dirty"] == "false"

    def test_a_change_not_committed_marks_the_run_dirty(
        self,
        tracker: MlflowTracker[list[str], list[str], list[str]],
        reader: MlflowClient,
        experiment_id: str,
        dataset: EvalDataset[list[str], list[str]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A run scored on uncommitted code says so: its commit alone would not name the code."""
        commit = self.checkout(tmp_path)
        (tmp_path / "suite.py").write_text("SCORED = False\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        tracker.evaluate(dataset)
        (run,) = reader.search_runs([experiment_id])

        assert run.data.tags["git.commit"] == commit
        assert run.data.tags["git.dirty"] == "true"

    def test_each_label_is_a_tag_of_its_own(self, run: Run, reader: MlflowClient) -> None:
        """A suite's labels filter its runs across experiments, as ``tags.<label> = "true"``."""
        found = reader.search_runs(
            [run.info.experiment_id],
            filter_string='tags.ingestion = "true" and tags.matching = "true"',
        )
        assert len(found) == 1

    def test_the_experiment_is_tagged_with_its_kind_and_the_labels(
        self, run: Run, reader: MlflowClient
    ) -> None:
        """MLflow 3's UI opens it on the runs table, and its list shows the suite's labels."""
        tags = reader.get_experiment(run.info.experiment_id).tags
        assert tags["mlflow.experimentKind"] == "custom_model_development"
        assert tags["ingestion"] == "true"
        assert tags["matching"] == "true"

    def test_the_dataset_is_logged_as_an_input(self, run: Run) -> None:
        """The run names the dataset it scored, with its size."""
        assert run.inputs is not None
        logged = run.inputs.dataset_inputs[0].dataset
        assert logged.name == "mini_corpus"
        assert logged.profile == '{"num_rows": 2}'

    def test_the_input_digest_changes_only_with_the_cases(
        self,
        tracker: MlflowTracker[list[str], list[str], list[str]],
        reader: MlflowClient,
        experiment_id: str,
        dataset: EvalDataset[list[str], list[str]],
    ) -> None:
        """Two runs share a digest only when they scored the same cases."""
        tracker.evaluate(dataset)
        tracker.evaluate(dataset)
        tracker.evaluate(dataset.model_copy(update={"cases": dataset.cases[:1]}))
        runs = reader.search_runs([experiment_id])
        digests = [run.inputs.dataset_inputs[0].dataset.digest for run in runs if run.inputs]
        assert sorted(Counter(digests).values()) == [1, 2]

    @pytest.mark.parametrize("adapter", [AgentCallingAdapter()])
    def test_every_agent_call_is_a_trace_of_the_run(self, run: Run, reader: MlflowClient) -> None:
        """One agent call per case gives the run one trace per case."""
        traces = reader.search_traces(locations=[run.info.experiment_id], run_id=run.info.run_id)
        assert len(traces) == 2

    def test_mlflow_global_state_is_restored(
        self,
        tracker: MlflowTracker[list[str], list[str], list[str]],
        dataset: EvalDataset[list[str], list[str]],
    ) -> None:
        """The tracking URI a caller had set survives the evaluation."""
        before = mlflow.get_tracking_uri()
        tracker.evaluate(dataset)
        assert mlflow.get_tracking_uri() == before

    def test_a_failing_evaluation_is_recorded_as_failed(
        self,
        store: Path,
        reader: MlflowClient,
        experiment: str,
        experiment_id: str,
        dataset: EvalDataset[list[str], list[str]],
    ) -> None:
        """The error still reaches the suite, and the run it leaves says FAILED."""
        tracker = MlflowTracker(
            Evaluator(FailingAdapter(), [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]),
            MlflowTrackerConfig(experiment=experiment, store=store),
        )

        with pytest.raises(RuntimeError, match="adapter exploded"):
            tracker.evaluate(dataset)

        assert [run.info.status for run in reader.search_runs([experiment_id])] == ["FAILED"]

    def test_runs_of_one_suite_share_its_experiment(
        self,
        tracker: MlflowTracker[list[str], list[str], list[str]],
        reader: MlflowClient,
        experiment_id: str,
        dataset: EvalDataset[list[str], list[str]],
    ) -> None:
        """A second run lands in the experiment of the first, so the two can be charted."""
        tracker.evaluate(dataset)
        tracker.evaluate(dataset)
        runs = reader.search_runs([experiment_id])
        assert [run.info.status for run in runs] == ["FINISHED"] * 2
