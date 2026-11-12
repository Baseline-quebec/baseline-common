"""Unit tests for the per-metric gate, the corpus mean and the non-regression baseline."""

from typing import ClassVar

import pytest

from baseline_core.core.llm.evalkit.abstract_metric import MetricResult
from baseline_core.core.llm.evalkit.case import EvalDataset
from baseline_core.core.llm.evalkit.evaluator import EvalResult, Evaluator
from baseline_core.core.llm.evalkit.gating import BaselineGate, CorpusGate, MetricGate
from tests.core.llm.evalkit.conftest import (
    DetectionCounts,
    ExactMatchMetric,
    FailingAdapter,
    InputCopyingAdapter,
    ItemRecallMetric,
)


class TestMetricGate:
    """Holding every case to the threshold each of its metrics carries."""

    def test_a_run_clearing_every_threshold_is_silent(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """A system reproducing its input passes both metrics on both cases."""
        results = Evaluator(
            InputCopyingAdapter(),
            [
                ItemRecallMetric(ItemRecallMetric.PASS_MARK),
                ExactMatchMetric(ItemRecallMetric.PASS_MARK),
            ],
        ).evaluate(dataset)
        MetricGate([
            ItemRecallMetric(ItemRecallMetric.PASS_MARK),
            ExactMatchMetric(ItemRecallMetric.PASS_MARK),
        ]).assert_all_pass(results)

    def test_a_miss_names_the_case_the_metric_and_both_numbers(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """The message identifies what fell, by its label rather than its class name."""
        metrics = [ExactMatchMetric(ItemRecallMetric.PASS_MARK)]
        results = Evaluator(InputCopyingAdapter(drop=1), metrics).evaluate(dataset)

        with pytest.raises(AssertionError, match=r"first: Exactness 0.00 < threshold 0.90"):
            MetricGate(metrics).assert_all_pass(results)

    def test_every_metric_of_every_case_is_reported(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """The gate never stops at the first miss: two metrics down on two cases means four."""
        metrics = [
            ItemRecallMetric(ItemRecallMetric.PASS_MARK),
            ExactMatchMetric(ItemRecallMetric.PASS_MARK),
        ]
        results = Evaluator(InputCopyingAdapter(drop=1), metrics).evaluate(dataset)
        fallen = MetricGate(metrics).misses(results)
        assert [(miss.case, miss.metric) for miss in fallen] == [
            ("first", "Item Recall"),
            ("first", "Exactness"),
            ("second", "Item Recall"),
            ("second", "Exactness"),
        ]

    def test_each_metric_gates_at_its_own_threshold(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """Five metrics mean five thresholds: a lenient one passes what a strict one fails."""
        results = Evaluator(
            InputCopyingAdapter(drop=1), [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]
        ).evaluate(dataset)
        assert MetricGate([ItemRecallMetric(threshold=0.0)]).misses(results) == []
        assert MetricGate([ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).misses(results)

    def test_the_recorded_score_is_what_gates(self) -> None:
        """The gate reads what the run computed; it never re-scores the output itself."""
        result = ItemRecallMetric.make_result("first", ["a", "b"], ["a"])
        recorded = result.model_copy(
            update={
                "scores": {"ItemRecallMetric": MetricResult[None](score=1.0, reason="recorded")}
            }
        )
        MetricGate([ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).assert_all_pass([recorded])

    def test_a_metric_the_run_never_scored_is_loud(self) -> None:
        """Gating on a metric absent from the run is a wiring bug, not a silent pass."""
        with pytest.raises(KeyError):
            MetricGate([ExactMatchMetric(ItemRecallMetric.PASS_MARK)]).misses([
                ItemRecallMetric.make_result("first", ["a"], ["a"])
            ])

    def test_an_empty_run_raises(self) -> None:
        """An empty run is a broken suite, not a perfect score."""
        with pytest.raises(AssertionError, match="No cases to gate"):
            MetricGate([ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).assert_all_pass([])


class TestMetricGateDrops:
    """The completeness gate: a tolerated failure must not pass as a shorter run."""

    def test_a_complete_run_is_silent(self, dataset: EvalDataset[list[str], list[str]]) -> None:
        """Every case of the dataset came back, so there is nothing to report."""
        results = Evaluator(
            InputCopyingAdapter(), [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]
        ).evaluate(dataset)
        MetricGate([ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).assert_no_case_drops(
            results, dataset
        )

    def test_a_dropped_case_is_named(self, dataset: EvalDataset[list[str], list[str]]) -> None:
        """The message names the dataset, both counts, and which case never ran."""
        results = Evaluator(
            FailingAdapter(only="a"),
            [ItemRecallMetric(ItemRecallMetric.PASS_MARK)],
            tolerate_errors=True,
        ).evaluate(dataset)

        with pytest.raises(
            AssertionError, match=r"mini_corpus: 1 of 2 cases did not run: \['first'\]"
        ):
            MetricGate([ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).assert_no_case_drops(
                results, dataset
            )

    def test_an_empty_run_names_every_case(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """A run that lost everything is the loudest case, not a vacuous pass."""
        with pytest.raises(AssertionError, match=r"2 of 2 cases did not run"):
            MetricGate([ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).assert_no_case_drops(
                [], dataset
            )

    def test_an_empty_dataset_has_nothing_to_drop(self) -> None:
        """Nothing asked for means nothing missing: the gate stays silent."""
        empty = EvalDataset[list[str], list[str]](name="empty", cases=[])
        MetricGate([ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).assert_no_case_drops([], empty)


class TestMetricGateMean:
    """The corpus mean: what holds a dimension whose per-case floor has to stay low."""

    @staticmethod
    def gate(floor: float) -> MetricGate[list[str], list[str]]:
        """A gate holding Item Recall to the corpus mean the metric carries.

        Args:
            floor: The corpus mean Item Recall must clear.

        Returns:
            The gate, scored by Item Recall and gated on its mean.
        """
        return MetricGate([ItemRecallMetric(ItemRecallMetric.PASS_MARK, mean=floor)])

    @property
    def mixed_run(self) -> list[EvalResult[list[str], list[str], list[str]]]:
        """Two cases scoring 1.00 and 0.50, for a mean of 0.75."""
        return [
            ItemRecallMetric.make_result("a", ["x"], ["x"]),
            ItemRecallMetric.make_result("b", ["x", "y"], ["x"]),
        ]

    def test_a_mean_above_the_floor_passes(self) -> None:
        """One weak case is absorbed as long as the average holds."""
        self.gate(0.6).assert_means(self.mixed_run)

    def test_the_floor_is_inclusive(self) -> None:
        """A mean landing exactly on the floor is not a failure."""
        self.gate(0.75).assert_means(self.mixed_run)

    def test_too_many_weak_cases_fail_the_mean(self) -> None:
        """The message carries the mean, the case count, and every case's rounded value."""
        with pytest.raises(
            AssertionError, match=r"mean Item Recall 0.75 is below 0.85 over 2 cases"
        ):
            self.gate(0.85).assert_means(self.mixed_run)

    def test_no_case_is_an_error_rather_than_a_pass(self) -> None:
        """An empty run must never satisfy a floor by vacuity."""
        with pytest.raises(AssertionError, match="No cases to average Item Recall over"):
            self.gate(0.85).assert_means([])

    def test_a_gate_holding_no_mean_asserts_nothing(self) -> None:
        """Per-case thresholds are the default; a corpus mean is opt-in on the metric."""
        MetricGate([ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).assert_means(self.mixed_run)

    def test_only_a_metric_carrying_a_mean_is_averaged(self) -> None:
        """Exact averages 0.50 here, but carries no mean, so it cannot fail the gate."""
        metrics = [
            ItemRecallMetric(ItemRecallMetric.PASS_MARK, mean=0.6),
            ExactMatchMetric(ItemRecallMetric.PASS_MARK),
        ]
        MetricGate(metrics).assert_means(self.mixed_run)


class TestCorpusGate:
    """Holding a corpus fold to the floor of each number it gates."""

    GATE: ClassVar[CorpusGate] = CorpusGate({"coverage": 0.95, "precision": 0.45})

    def test_a_fold_clearing_every_floor_is_silent(self) -> None:
        """Both gated numbers above their floor raises nothing."""
        self.GATE.assert_clears(DetectionCounts(hits=20, detected=20, expected=20))

    def test_the_floor_is_inclusive(self) -> None:
        """A number landing exactly on its floor is not a shortfall."""
        CorpusGate({"recall": 0.5}).assert_clears(DetectionCounts(hits=1, detected=1, expected=2))

    def test_a_shortfall_names_the_field_and_both_numbers(self) -> None:
        """The message says which number fell, to what, and from what floor."""
        with pytest.raises(AssertionError, match=r"Corpus coverage 0\.50 is below 0\.95"):
            self.GATE.assert_clears(DetectionCounts(hits=1, detected=1, expected=2))

    def test_every_gated_number_is_reported(self) -> None:
        """The gate never stops at the first shortfall: two down means two clauses."""
        fallen = self.GATE.shortfalls(DetectionCounts(hits=1, detected=10, expected=10))
        assert [(short.field, short.floor) for short in fallen] == [
            ("coverage", 0.95),
            ("precision", 0.45),
        ]

    def test_an_ungated_number_cannot_fail_the_fold(self) -> None:
        """A number this gate holds no floor for is not its business."""
        CorpusGate({"recall": 0.0}).assert_clears(
            DetectionCounts(hits=0, detected=10, expected=10)
        )

    def test_a_floor_on_a_number_the_fold_never_names_is_loud(self) -> None:
        """Gating a number the scorer does not report is a wiring bug, not a silent pass."""
        with pytest.raises(KeyError):
            CorpusGate({"f1": 0.5}).shortfalls(DetectionCounts(hits=1, detected=1, expected=1))


class TestBaselineGate:
    """Freezing the gated numbers of a run, then comparing them to a reference."""

    REFERENCE: ClassVar[dict[str, dict[str, float]]] = {
        "kept": {"precision": 0.5, "recall": 0.95, "coverage": 1.0},
        "dropped": {"precision": 0.5, "recall": 0.9, "coverage": 0.9},
    }
    GATED: ClassVar[tuple[str, ...]] = ("recall", "coverage")
    DIGITS: ClassVar[int] = 3
    TOL: ClassVar[float] = 0.005

    @property
    def report(self) -> BaselineGate:
        """A report gating recall and coverage at the project's precision and tolerance."""
        return BaselineGate(self.GATED, digits=self.DIGITS, tol=self.TOL)

    def test_properties_are_read_and_rounded(self) -> None:
        """Derived properties are fair game, and rounding kills float noise."""
        gated = BaselineGate(("precision", "recall"), digits=self.DIGITS, tol=self.TOL)
        assert gated.values(DetectionCounts(hits=2, detected=3, expected=3)) == {
            "precision": 0.667,
            "recall": 0.667,
        }

    def test_the_table_keys_by_case(self) -> None:
        """The per-case table is exactly what a reference file freezes."""
        table = self.report.table({"a": DetectionCounts(hits=1, detected=1, expected=2)})
        assert table == {"a": {"recall": 0.5, "coverage": 0.5}}

    def test_a_matching_run_reports_nothing(self) -> None:
        """Equal numbers are not a regression."""
        assert self.report.regressions(self.REFERENCE, self.REFERENCE).root == []

    def test_a_drop_beyond_tolerance_is_reported(self) -> None:
        """Recall below the reference minus the tolerance fails, and says by how much."""
        current = {"kept": {"recall": 0.9, "coverage": 1.0}}
        worse = self.report.regressions(current, self.REFERENCE).root
        assert [(r.case, r.field) for r in worse] == [("kept", "recall")]
        assert str(worse[0]) == "kept: recall 0.90 < reference 0.95"

    def test_a_drop_within_tolerance_is_absorbed(self) -> None:
        """Rounding noise under the tolerance must not fail a run."""
        current = {"kept": {"recall": 0.947, "coverage": 1.0}}
        assert self.report.regressions(current, self.REFERENCE).root == []

    def test_an_improvement_is_not_a_regression(self) -> None:
        """Scoring above the reference is the point of the exercise."""
        current = {"kept": {"recall": 1.0, "coverage": 1.0}}
        assert self.report.regressions(current, self.REFERENCE).root == []

    def test_every_gated_field_is_checked(self) -> None:
        """Two fields down means two entries: the report never stops at the first."""
        current = {"dropped": {"recall": 0.5, "coverage": 0.5}}
        worse = self.report.regressions(current, self.REFERENCE).root
        assert [r.field for r in worse] == ["recall", "coverage"]

    def test_an_ungated_field_is_ignored(self) -> None:
        """Precision is in the reference but not gated here, so its collapse passes."""
        current = {"kept": {"precision": 0.0, "recall": 0.95, "coverage": 1.0}}
        assert self.report.regressions(current, self.REFERENCE).root == []

    def test_a_case_absent_from_the_reference_is_skipped(self) -> None:
        """A brand-new case has nothing to regress against: it is skipped, not failed."""
        current = {"unknown": {"recall": 0.0, "coverage": 0.0}}
        assert self.report.regressions(current, self.REFERENCE).root == []

    def test_a_field_absent_from_the_reference_is_skipped(self) -> None:
        """A field the frozen reference never recorded cannot gate this run."""
        reference = {"kept": {"recall": 0.95}}
        current = {"kept": {"recall": 0.95, "coverage": 0.0}}
        assert self.report.regressions(current, reference).root == []

    def test_the_tolerance_is_constructor_state(self) -> None:
        """A tighter tolerance catches what the project's own absorbs."""
        current = {"kept": {"recall": 0.947, "coverage": 1.0}}
        strict = BaselineGate(self.GATED, digits=self.DIGITS, tol=0.0)
        assert strict.regressions(current, self.REFERENCE).root

    def test_regressions_describe_themselves_as_one_line(self) -> None:
        """The formatted message names every fallen case, for the assertion that raises it."""
        current = {"dropped": {"recall": 0.5, "coverage": 0.9}}
        worse = self.report.regressions(current, self.REFERENCE)
        assert str(worse) == "dropped: recall 0.50 < reference 0.90"

    def test_nothing_to_describe_is_the_empty_string(self) -> None:
        """A clean run formats to nothing at all."""
        assert not str(self.report.regressions({}, self.REFERENCE))

    def test_it_reproduces_the_project_baseline_gate(self) -> None:
        """End to end on score objects: gate recall/coverage, skip the unknown case."""
        scores = {
            "kept": DetectionCounts(hits=18, detected=36, expected=20),
            "unknown": DetectionCounts(hits=0, detected=1, expected=10),
        }
        worse = self.report.regressions(self.report.table(scores), self.REFERENCE).root
        assert [(r.case, r.field, r.value) for r in worse] == [
            ("kept", "recall", 0.9),
            ("kept", "coverage", 0.9),
        ]
