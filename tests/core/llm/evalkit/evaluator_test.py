"""Unit tests for the evaluator: order, score keys, the cost envelope and error tolerance."""

import pytest

from baseline_core.core.llm.evalkit.case import EvalDataset
from baseline_core.core.llm.evalkit.eval_run import RunCost
from baseline_core.core.llm.evalkit.evaluator import Evaluator
from tests.core.llm.evalkit.conftest import (
    ExactMatchMetric,
    FailingAdapter,
    InFlightCountingAdapter,
    InputCopyingAdapter,
    ItemRecallMetric,
    TokenBillingAdapter,
)


class TestEvaluator:
    """``evaluate``: one result per case, in dataset order, scored by every bound metric."""

    def test_every_case_is_evaluated_in_dataset_order(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """Results follow the dataset order and carry their case."""
        results = Evaluator(
            InputCopyingAdapter(), [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]
        ).evaluate(dataset)
        assert [result.case.name for result in results] == ["first", "second"]

    def test_the_adapter_receives_the_case_input_only(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """The adapter sees the inputs, never the case or its expected value."""
        adapter = InputCopyingAdapter()
        Evaluator(adapter, [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).evaluate(dataset)
        assert sorted(adapter.seen) == [["a", "b"], ["c"]]

    def test_scores_are_keyed_by_metric_class_name(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """Each metric gets its own slot, keyed by its class name."""
        evaluator = Evaluator(
            InputCopyingAdapter(),
            [
                ItemRecallMetric(ItemRecallMetric.PASS_MARK),
                ExactMatchMetric(ItemRecallMetric.PASS_MARK),
            ],
        )
        assert sorted(evaluator.evaluate(dataset)[0].scores) == [
            "ExactMatchMetric",
            "ItemRecallMetric",
        ]

    def test_two_metrics_of_one_class__are_refused(self) -> None:
        """Their scores share a key: the second would overwrite the first."""
        metrics = [ItemRecallMetric(ItemRecallMetric.PASS_MARK), ItemRecallMetric(0.9)]

        with pytest.raises(ValueError, match="ItemRecallMetric"):
            Evaluator(InputCopyingAdapter(), metrics)

    def test_metrics_score_the_output_against_the_expected_value(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """A system dropping its last item scores a partial overlap and a failed exactness."""
        evaluator = Evaluator(
            InputCopyingAdapter(drop=1),
            [
                ItemRecallMetric(ItemRecallMetric.PASS_MARK),
                ExactMatchMetric(ItemRecallMetric.PASS_MARK),
            ],
        )
        results = evaluator.evaluate(dataset)
        assert results[0].scores["ItemRecallMetric"].score == 0.5
        assert results[0].scores["ExactMatchMetric"].score == 0.0
        assert results[0].output.value == ["a"]

    def test_a_deterministic_adapter_leaves_the_cost_unset(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """The envelope's three cost fields stay ``None`` for a system that bills nothing."""
        results = Evaluator(
            InputCopyingAdapter(), [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]
        ).evaluate(dataset)
        envelope = results[0].output
        assert (envelope.input_tokens, envelope.output_tokens, envelope.elapsed_s) == (
            None,
            None,
            None,
        )

    def test_a_metered_adapter_keeps_its_cost(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """A billing system reports tokens and wall time through the same contract."""
        results = Evaluator(
            TokenBillingAdapter(), [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]
        ).evaluate(dataset)
        assert results[0].output.input_tokens == 2
        assert results[0].output.output_tokens == 4
        assert RunCost.total(results).input_tokens == 3

    def test_an_empty_dataset_produces_no_result(self) -> None:
        """Nothing to evaluate means nothing to report, not a failure."""
        empty = EvalDataset[list[str], list[str]](name="empty", cases=[])
        assert (
            Evaluator(
                InputCopyingAdapter(), [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]
            ).evaluate(empty)
            == []
        )

    def test_the_cases_are_all_in_flight_at_once(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """The evaluator throttles nothing of its own: both cases start before either answers."""
        adapter = InFlightCountingAdapter()
        Evaluator(adapter, [ItemRecallMetric(ItemRecallMetric.PASS_MARK)]).evaluate(dataset)
        assert adapter.peak == 2


class TestEvaluatorErrorTolerance:
    """Whether one case's failure sinks the whole evaluation or is dropped from it."""

    def test_a_failing_case_sinks_the_evaluation_by_default(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """An unexpected error is a bug until the caller says otherwise."""
        evaluator = Evaluator(FailingAdapter(), [ItemRecallMetric(ItemRecallMetric.PASS_MARK)])

        with pytest.raises(RuntimeError, match="adapter exploded"):
            evaluator.evaluate(dataset)

    def test_a_tolerated_failure_drops_only_its_own_case(
        self, dataset: EvalDataset[list[str], list[str]]
    ) -> None:
        """A rate limit on one document must not discard what the others already measured."""
        evaluator = Evaluator(
            FailingAdapter(only="a"),
            [ItemRecallMetric(ItemRecallMetric.PASS_MARK)],
            tolerate_errors=True,
        )
        assert [result.case.name for result in evaluator.evaluate(dataset)] == ["second"]
