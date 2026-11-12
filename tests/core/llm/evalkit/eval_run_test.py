"""Unit tests for run persistence and cost totals."""

from datetime import timedelta
from pathlib import Path

from baseline_core.core.llm.evalkit.case import EvalDataset
from baseline_core.core.llm.evalkit.eval_run import EvalRun, RunCost
from baseline_core.core.llm.evalkit.evaluator import EvalResult
from tests.core.llm.evalkit.conftest import ItemRecallMetric


type Results = list[EvalResult[list[str], list[str], list[str]]]
type MiniRun = EvalRun[list[str], list[str], list[str], None]


class TestRunCost:
    """Cost totals: summed when reported, absent when nothing reported them."""

    def test_reported_costs_are_summed(self, cheap_run: Results) -> None:
        """Both cases billed, so every total is the sum of its field."""
        cost = RunCost.total(cheap_run)
        assert (cost.input_tokens, cost.output_tokens, cost.elapsed_s) == (16, 6, 2.0)
        assert cost.reported

    def test_a_cost_free_run_reports_nothing(self) -> None:
        """A deterministic run leaves every total unset rather than claiming zero."""
        cost = RunCost.total([ItemRecallMetric.make_result("only", ["a"], ["a"])])
        assert (cost.input_tokens, cost.output_tokens, cost.elapsed_s) == (None, None, None)
        assert not cost.reported

    def test_a_partly_billed_run_sums_what_it_has(self, cheap_run: Results) -> None:
        """One billed case among two is enough to report that field, and only that field."""
        cost = RunCost.total([
            ItemRecallMetric.make_result("a", ["a"], ["a"], input_tokens=7),
            cheap_run[0],
        ])
        assert cost.input_tokens == 17
        assert cost.elapsed_s == 1.5

    def test_an_empty_run_reports_nothing(self) -> None:
        """No case means no total, and no crash."""
        assert not RunCost.total([]).reported


class TestEvalRun:
    """Assembling a run report — timestamp, cost totals, corpus fold — and persisting it."""

    @staticmethod
    def mean_score(results: Results) -> float:
        """The corpus micro-average a caller folds before handing it to the run."""
        return sum(r.scores["ItemRecallMetric"].score for r in results) / len(results)

    def test_the_run_is_timestamped_in_utc(
        self, dataset: EvalDataset[list[str], list[str]], cheap_run: Results
    ) -> None:
        """The stamp is timezone-aware, so two runs are comparable across machines."""
        run: MiniRun = EvalRun.build(dataset, cheap_run)
        assert run.ran_at.tzinfo is not None
        assert run.ran_at.utcoffset() == timedelta(0)

    def test_the_run_is_named_after_the_dataset_it_scored(
        self, dataset: EvalDataset[list[str], list[str]], cheap_run: Results
    ) -> None:
        """The name comes off the dataset, so it cannot drift from what was actually run."""
        run: MiniRun = EvalRun.build(dataset, cheap_run)
        assert run.dataset == dataset.name
        assert len(run.results) == 2
        assert run.cost.input_tokens == 16

    def test_without_a_summary_the_field_stays_unset(
        self, dataset: EvalDataset[list[str], list[str]], cheap_run: Results
    ) -> None:
        """A run needs no corpus fold: the summary stays unset."""
        assert EvalRun.build(dataset, cheap_run).summary is None

    def test_the_summary_is_the_fold_the_caller_computed(
        self, dataset: EvalDataset[list[str], list[str]], cheap_run: Results
    ) -> None:
        """A corpus micro-average is folded by the caller and lands in the summary."""
        results = cheap_run
        assert EvalRun.build(dataset, results, self.mean_score(results)).summary == 0.5

    def test_the_file_is_named_after_the_dataset_and_the_stamp(
        self, dataset: EvalDataset[list[str], list[str]], tmp_path: Path, cheap_run: Results
    ) -> None:
        """The name carries the dataset, so a folder can hold several corpora."""
        path = EvalRun.build(dataset, cheap_run).write(tmp_path / "runs")
        assert path.parent == tmp_path / "runs"
        assert path.name.startswith("mini_corpus_")
        assert path.suffix == ".json"

    def test_two_runs_in_the_same_second__keep_their_own_file(
        self, dataset: EvalDataset[list[str], list[str]], tmp_path: Path, cheap_run: Results
    ) -> None:
        """A second run stamped alike never overwrites the first."""
        run: MiniRun = EvalRun.build(dataset, cheap_run)

        first = run.write(tmp_path)
        second = run.write(tmp_path)

        assert first != second
        assert first.name == f"mini_corpus_{run.ran_at:%Y%m%dT%H%M%S}Z.json"
        assert second.name == f"mini_corpus_{run.ran_at:%Y%m%dT%H%M%S}Z_2.json"
        assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")

    def test_the_persisted_run_reloads(
        self, dataset: EvalDataset[list[str], list[str]], tmp_path: Path, cheap_run: Results
    ) -> None:
        """A written run round-trips through its parametrized model, scores included."""
        path = EvalRun.build(dataset, cheap_run).write(tmp_path)
        run = EvalRun[list[str], list[str], list[str], None].model_validate_json(
            path.read_text(encoding="utf-8")
        )
        assert [result.case.name for result in run.results] == ["first", "second"]
        assert run.results[1].scores["ItemRecallMetric"].reason == "0/1 matched"
        assert run.cost.output_tokens == 6
