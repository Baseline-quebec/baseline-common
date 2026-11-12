"""Unit tests for the metric contract: the threshold verdict and the typed detail."""

from typing import ClassVar

from baseline_core.core.llm.evalkit.abstract_metric import MetricResult
from tests.core.llm.evalkit.conftest import ExactMatchMetric, ItemRecall, ItemRecallMetric


class TestMetric:
    """The kit's ``AbstractMetric``: a pure score plus a threshold verdict its suite chooses."""

    BELOW: ClassVar[float] = 0.89

    def test_the_threshold_is_the_one_the_suite_passed(self) -> None:
        """The kit declares no default: a metric gates at exactly what it was built with."""
        assert ItemRecallMetric(ItemRecallMetric.PASS_MARK).threshold == ItemRecallMetric.PASS_MARK

    def test_score_at_the_threshold_passes(self) -> None:
        """The threshold is inclusive: scoring exactly it is a pass."""
        metric = ItemRecallMetric(ItemRecallMetric.PASS_MARK)
        result = MetricResult[None](score=ItemRecallMetric.PASS_MARK, reason="on the line")
        assert metric.passes(result)

    def test_score_below_the_threshold_fails(self) -> None:
        """A score under the threshold is a fail, whatever its reason says."""
        metric = ItemRecallMetric(ItemRecallMetric.PASS_MARK)
        assert not metric.passes(MetricResult[None](score=self.BELOW, reason="close"))

    def test_a_lower_threshold_moves_the_verdict(self) -> None:
        """The constructor threshold, and nothing else, decides the verdict."""
        assert ItemRecallMetric(threshold=0.5).passes(
            MetricResult[None](score=self.BELOW, reason="close")
        )

    def test_scoring_is_pure_and_reentrant(self) -> None:
        """Scoring twice returns equal results and leaves no state on the metric."""
        metric = ItemRecallMetric(ItemRecallMetric.PASS_MARK)
        first = metric.score(["a", "b"], ["a"])
        second = metric.score(["a", "b"], ["a"])
        assert first == second
        assert not hasattr(metric, "success")

    def test_detail_keeps_its_type(self) -> None:
        """The typed payload survives scoring, so a rich score is not flattened to a float."""
        result = ItemRecallMetric(ItemRecallMetric.PASS_MARK).score(["a", "b"], ["a"])
        assert result.score == 0.5
        assert result.detail == ItemRecall(hits=1, missed=["b"])

    def test_detail_defaults_to_none(self) -> None:
        """A metric with nothing to add leaves the detail unset."""
        assert ExactMatchMetric(ItemRecallMetric.PASS_MARK).score(["a"], ["a"]).detail is None

    def test_the_label_drops_the_class_suffix(self) -> None:
        """Without a declared name, the label is the class name minus the ``Metric`` word."""
        assert ItemRecallMetric(ItemRecallMetric.PASS_MARK).label == "Item Recall"

    def test_an_explicit_display_name_wins(self) -> None:
        """A metric declaring its label keeps it, whatever its class is called."""
        assert ExactMatchMetric(ItemRecallMetric.PASS_MARK).label == "Exactness"
