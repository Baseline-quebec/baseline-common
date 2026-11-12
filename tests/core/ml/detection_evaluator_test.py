"""Tests for ``LabelMetrics``, ``DetectionReport`` and ``DetectionEvaluator``."""

from pathlib import Path
from typing import ClassVar

import pytest

from baseline_core.core.ml.detection_evaluator import (
    DetectionEvaluator,
    DetectionReport,
    LabelMetrics,
)
from baseline_core.core.ml.modality.image.annotation import (
    BoundingBox,
    BoxAnnotation,
    ImageAnnotation,
)


class TestLabelMetrics:
    """Tests for ``LabelMetrics``."""

    def test__from_counts__should_derive_rates(self) -> None:
        """Precision and recall follow the counts."""
        metrics = LabelMetrics.from_counts(tp=3, fp=1, fn=1)

        assert (metrics.precision, metrics.recall, metrics.truth_count) == (0.75, 0.75, 4)

    def test__from_counts__when_nothing__should_leave_rates_undefined(self) -> None:
        """No prediction and no truth give no rate, not a division error."""
        metrics = LabelMetrics.from_counts(tp=0, fp=0, fn=0)

        assert (metrics.precision, metrics.recall) == (None, None)


class TestDetectionReport:
    """Tests for ``DetectionReport``."""

    @pytest.fixture
    def report(self) -> DetectionReport:
        """A report with one confusion."""
        return DetectionReport(
            images=2,
            iou_threshold=0.5,
            map=0.5,
            map_50=0.75,
            per_label={
                "box": LabelMetrics.from_counts(tp=1, fp=1, fn=0),
                "bag": LabelMetrics.from_counts(tp=0, fp=0, fn=1),
            },
            confusions={"bag": {"box": 1}},
        )

    def test__to_markdown__should_render_label_and_confusion_tables(
        self, report: DetectionReport
    ) -> None:
        """Both tables appear, undefined rates as ``n/a``."""
        markdown = report.to_markdown()

        assert "mAP: 0.500, mAP@50: 0.750" in markdown
        assert "| box | 1 | 1 | 1 | 0 | 0.500 | 1.000 |" in markdown
        assert "| bag | 1 | 0 | 0 | 1 | n/a | 0.000 |" in markdown
        assert "| bag | box | 1 |" in markdown

    def test__save__then_load__should_round_trip(
        self, report: DetectionReport, tmp_path: Path
    ) -> None:
        """The JSON file reads back to the same report."""
        path = tmp_path / "out" / "report.json"

        report.save(path)

        assert DetectionReport.model_validate_json(path.read_text()) == report


class TestDetectionEvaluator:
    """Tests for ``DetectionEvaluator`` on hand-built predictions."""

    HIGH_SCORE: ClassVar[float] = 0.9
    LOW_SCORE: ClassVar[float] = 0.2

    @pytest.fixture
    def evaluator(self) -> DetectionEvaluator:
        """Default thresholds: IoU 0.5, every score kept."""
        return DetectionEvaluator()

    @pytest.fixture
    def predictions(self) -> list[ImageAnnotation]:
        """Box: hit. Mixed: bag taken for box, bag missed. Empty: spurious bag."""
        return [
            ImageAnnotation(
                file_name="box.png",
                width=64,
                height=48,
                boxes=[
                    BoxAnnotation(
                        label="box",
                        box=BoundingBox(x1=4, y1=4, x2=36, y2=28),
                        score=self.HIGH_SCORE,
                    )
                ],
            ),
            ImageAnnotation(
                file_name="mixed.png",
                width=64,
                height=48,
                boxes=[
                    BoxAnnotation(
                        label="box",
                        box=BoundingBox(x1=0, y1=0, x2=20, y2=20),
                        score=self.HIGH_SCORE,
                    ),
                    BoxAnnotation(
                        label="box",
                        box=BoundingBox(x1=30, y1=10, x2=60, y2=40),
                        score=self.LOW_SCORE,
                    ),
                ],
            ),
            ImageAnnotation(
                file_name="empty.png",
                width=64,
                height=48,
                boxes=[
                    BoxAnnotation(
                        label="bag", box=BoundingBox(x1=1, y1=1, x2=9, y2=9), score=self.LOW_SCORE
                    )
                ],
            ),
        ]

    def test__evaluate__should_count_matches_per_label_and_confusions(
        self,
        evaluator: DetectionEvaluator,
        predictions: list[ImageAnnotation],
        annotated_images: list[ImageAnnotation],
    ) -> None:
        """Greedy IoU matching: hits, a mislabeled overlap, a miss and a spurious box."""
        report = evaluator.evaluate(predictions, annotated_images)

        assert report.images == len(annotated_images)
        assert report.per_label["box"] == LabelMetrics.from_counts(tp=2, fp=1, fn=0)
        assert report.per_label["bag"] == LabelMetrics.from_counts(tp=0, fp=1, fn=1)
        assert report.confusions == {"bag": {"box": 1}}
        assert 0.0 < report.map_50 <= 1.0
        assert report.map <= report.map_50

    def test__evaluate__when_score_threshold__should_ignore_low_predictions(
        self, predictions: list[ImageAnnotation], annotated_images: list[ImageAnnotation]
    ) -> None:
        """Below the threshold, the mislabeled box and the spurious box disappear."""
        evaluator = DetectionEvaluator(score_threshold=self.HIGH_SCORE)

        report = evaluator.evaluate(predictions, annotated_images)

        assert report.per_label["box"].fp == 0
        assert report.per_label["bag"] == LabelMetrics.from_counts(tp=0, fp=0, fn=1)
        assert report.confusions == {}

    def test__evaluate__when_images_differ__should_raise_value_error(
        self,
        evaluator: DetectionEvaluator,
        predictions: list[ImageAnnotation],
        annotated_images: list[ImageAnnotation],
    ) -> None:
        """Both sides are named, so the mismatch is easy to trace."""
        partial = predictions[:1]

        with pytest.raises(ValueError, match=r"only in truth: \['empty.png', 'mixed.png'\]"):
            evaluator.evaluate(partial, annotated_images)

    def test__evaluate__when_prediction_has_no_score__should_raise_value_error(
        self, evaluator: DetectionEvaluator, annotated_images: list[ImageAnnotation]
    ) -> None:
        """Ground truth passed as predictions is a caller mistake."""
        with pytest.raises(ValueError, match="without a score"):
            evaluator.evaluate(annotated_images, annotated_images)

    def test__evaluate__when_label_predicted_but_absent_from_truth__should_count_false_positives(
        self, evaluator: DetectionEvaluator, empty_image: ImageAnnotation
    ) -> None:
        """A label the truth never uses still gets an id, so mAP does not fail on it."""
        prediction = empty_image.model_copy(
            update={
                "boxes": [
                    BoxAnnotation(
                        label="sauce",
                        box=BoundingBox(x1=1, y1=1, x2=9, y2=9),
                        score=self.HIGH_SCORE,
                    )
                ]
            }
        )

        report = evaluator.evaluate([prediction], [empty_image])

        assert report.per_label == {"sauce": LabelMetrics.from_counts(tp=0, fp=1, fn=0)}

    def test__evaluate__when_file_repeated__should_raise_value_error(
        self, evaluator: DetectionEvaluator, predictions: list[ImageAnnotation]
    ) -> None:
        """Two entries for one file cannot be matched to the truth."""
        with pytest.raises(ValueError, match=r"more than once: \['box.png'\]"):
            evaluator.evaluate([*predictions, predictions[0]], predictions)
