"""Comparison of predicted and ground-truth annotations: mAP, per-label counts, confusions."""

from collections import Counter, defaultdict
from pathlib import Path

import torch
from pydantic import BaseModel, Field
from torchmetrics.detection import MeanAveragePrecision

from baseline_core.core.ml.modality.image.annotation import BoxAnnotation, ImageAnnotation


class LabelMetrics(BaseModel):
    """Detection counts of one label over a whole set.

    Attributes:
        tp: Predictions matching a ground-truth box of the same label.
        fp: Predictions matching nothing, or a box of another label.
        fn: Ground-truth boxes no prediction of the same label matched.
        precision: ``tp / (tp + fp)``, or ``None`` without any prediction.
        recall: ``tp / (tp + fn)``, or ``None`` without any ground-truth box.
        truth_count: Ground-truth boxes of the label.
    """

    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    precision: float | None
    recall: float | None
    truth_count: int = Field(ge=0)

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> "LabelMetrics":
        """Derive the rates from the counts.

        Args:
            tp: True positives.
            fp: False positives.
            fn: False negatives.

        Returns:
            The metrics, with undefined rates left as ``None``.
        """
        return cls(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            truth_count=tp + fn,
        )


class DetectionReport(BaseModel):
    """How a set of predictions compares with the ground truth.

    Attributes:
        images: Number of images compared.
        iou_threshold: Minimum IoU for a prediction to match a ground-truth box.
        map: Mean average precision over IoU thresholds 0.5 to 0.95.
        map_50: Mean average precision at IoU 0.5.
        per_label: Counts and rates of every label.
        confusions: Per truth label, how often an overlapping prediction named each other label.
    """

    images: int = Field(ge=0)
    iou_threshold: float
    map: float
    map_50: float
    per_label: dict[str, LabelMetrics]
    confusions: dict[str, dict[str, int]]

    def to_markdown(self) -> str:
        """Render the report as Markdown tables."""
        lines = [
            f"Images: {self.images}, IoU threshold: {self.iou_threshold}",
            "",
            f"mAP: {self.map:.3f}, mAP@50: {self.map_50:.3f}",
            "",
            "| Label | Truth | TP | FP | FN | Precision | Recall |",
            "|---|---|---|---|---|---|---|",
        ]
        lines.extend(
            f"| {label} | {m.truth_count} | {m.tp} | {m.fp} | {m.fn} "
            f"| {self._rate(m.precision)} | {self._rate(m.recall)} |"
            for label, m in self.per_label.items()
        )

        if self.confusions:
            lines.extend(["", "| Truth | Predicted as | Count |", "|---|---|---|"])
            lines.extend(
                f"| {truth} | {predicted} | {count} |"
                for truth, predicted_counts in self.confusions.items()
                for predicted, count in predicted_counts.items()
            )

        return "\n".join(lines) + "\n"

    def save(self, path: Path) -> None:
        """Write the report as JSON, creating parent folders.

        Args:
            path: Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _rate(value: float | None) -> str:
        """Format a rate with three decimals, ``n/a`` when undefined."""
        return "n/a" if value is None else f"{value:.3f}"


class DetectionEvaluator:
    """Compares predictions with the ground truth, image by image.

    Attributes:
        iou_threshold: Minimum IoU for a prediction to match a ground-truth box.
        score_threshold: Predictions scoring below are ignored.
    """

    def __init__(self, iou_threshold: float = 0.5, score_threshold: float = 0.0) -> None:
        """Store the thresholds.

        Args:
            iou_threshold: Minimum IoU for a prediction to match a ground-truth box.
            score_threshold: Predictions scoring below are ignored.
        """
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def evaluate(
        self, predictions: list[ImageAnnotation], truth: list[ImageAnnotation]
    ) -> DetectionReport:
        """Compare ``predictions`` with ``truth``, matched by file name.

        Args:
            predictions: Predicted boxes, with scores.
            truth: Ground-truth boxes of the labels the model learns.

        Returns:
            The report, covering every label of either side in name order.
        """
        predicted_by_file = self.by_file_name(predictions)
        truth_by_file = self.by_file_name(truth)
        self.check_same_files(predicted_by_file, truth_by_file)
        labels = sorted({box.label for image in [*predictions, *truth] for box in image.boxes})
        label2id = {label: index for index, label in enumerate(labels)}
        mean_ap = MeanAveragePrecision(box_format="xyxy")
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        confusions: dict[str, Counter[str]] = defaultdict(Counter)

        for truth_image in truth:
            predicted_image = predicted_by_file[truth_image.file_name]
            kept = self.scored_boxes(predicted_image)
            mean_ap.update(
                [self._metric_target(kept, label2id, with_scores=True)],
                [self._metric_target(truth_image.boxes, label2id, with_scores=False)],
            )
            self.match_image(kept, truth_image.boxes, counts, confusions)

        mean_ap_values = mean_ap.compute()
        return DetectionReport(
            images=len(truth),
            iou_threshold=self.iou_threshold,
            map=float(mean_ap_values["map"]),
            map_50=float(mean_ap_values["map_50"]),
            per_label={
                label: LabelMetrics.from_counts(
                    counts[label]["tp"], counts[label]["fp"], counts[label]["fn"]
                )
                for label in labels
            },
            confusions={label: dict(mistaken) for label, mistaken in confusions.items()},
        )

    @staticmethod
    def by_file_name(images: list[ImageAnnotation]) -> dict[str, ImageAnnotation]:
        """Index ``images`` by file name.

        Raises:
            ValueError: Listing the file names given more than once.
        """
        file_name_counts = Counter(image.file_name for image in images)
        repeated = sorted(name for name, count in file_name_counts.items() if count > 1)

        if repeated:
            error_msg = f"Images given more than once: {repeated}"
            raise ValueError(error_msg)

        return {image.file_name: image for image in images}

    @staticmethod
    def check_same_files(
        predicted_by_file: dict[str, ImageAnnotation], truth_by_file: dict[str, ImageAnnotation]
    ) -> None:
        """Raise if predictions and truth do not cover the same files.

        Raises:
            ValueError: Listing the files present on one side only.
        """
        predicted_files = set(predicted_by_file)
        truth_files = set(truth_by_file)

        if predicted_files != truth_files:
            error_msg = (
                f"Images only in predictions: {sorted(predicted_files - truth_files)}; "
                f"only in truth: {sorted(truth_files - predicted_files)}."
            )
            raise ValueError(error_msg)

    def scored_boxes(self, image: ImageAnnotation) -> list[BoxAnnotation]:
        """The predictions of ``image`` at or above ``score_threshold``, best first.

        Raises:
            ValueError: If a box has no score: it is not a prediction.
        """
        unscored = [box for box in image.boxes if box.score is None]

        if unscored:
            error_msg = f"Predictions without a score in {image.file_name}: {len(unscored)}."
            raise ValueError(error_msg)

        kept = [
            box
            for box in image.boxes
            if box.score is not None and box.score >= self.score_threshold
        ]
        return sorted(kept, key=lambda box: box.score or 0.0, reverse=True)

    def match_image(
        self,
        predicted: list[BoxAnnotation],
        truth: list[BoxAnnotation],
        counts: dict[str, Counter[str]],
        confusions: dict[str, Counter[str]],
    ) -> None:
        """Match each prediction to the unmatched truth box it overlaps most, and count.

        Args:
            predicted: Predictions, best score first.
            truth: Ground-truth boxes.
            counts: ``tp``, ``fp`` and ``fn`` per label, updated in place.
            confusions: Truth label to predicted label counts, updated in place.
        """
        unmatched = list(range(len(truth)))

        for box in predicted:
            best_index, best_iou = None, self.iou_threshold

            for index in unmatched:
                iou = box.box.iou(truth[index].box)

                if iou >= best_iou:
                    best_index, best_iou = index, iou

            if best_index is None:
                counts[box.label]["fp"] += 1
                continue

            matched = truth[best_index]

            if matched.label == box.label:
                unmatched.remove(best_index)
                counts[box.label]["tp"] += 1
            else:
                counts[box.label]["fp"] += 1
                confusions[matched.label][box.label] += 1

        for index in unmatched:
            counts[truth[index].label]["fn"] += 1

    @staticmethod
    def _metric_target(
        boxes: list[BoxAnnotation], label2id: dict[str, int], *, with_scores: bool
    ) -> dict[str, torch.Tensor]:
        """Encode ``boxes`` as a ``torchmetrics`` mean average precision target."""
        target = {
            "boxes": torch.tensor([box.box.xyxy for box in boxes]).reshape(-1, 4),
            "labels": torch.tensor([label2id[box.label] for box in boxes], dtype=torch.long),
        }

        if with_scores:
            target["scores"] = torch.tensor([box.score or 0.0 for box in boxes])

        return target
