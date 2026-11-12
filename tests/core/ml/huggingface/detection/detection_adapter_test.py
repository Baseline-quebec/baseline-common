"""Tests for ``DetectionAdapter``."""

from typing import ClassVar

import pytest
import torch

from baseline_core.core.ml.huggingface.detection.detection_adapter import DetectionAdapter
from baseline_core.core.ml.modality.image.annotation import ImageAnnotation


class TestDetectionAdapter:
    """Tests for ``DetectionAdapter`` over ``box`` and ``bag``."""

    IMAGE_ID: ClassVar[int] = 3

    @pytest.fixture
    def result(self) -> dict[str, torch.Tensor]:
        """A post-processed result: a zero-height ``box`` and a scored ``bag`` box."""
        return {
            "boxes": torch.tensor([[1.0, 5.0, 10.0, 5.0], [1.0, 1.0, 10.0, 10.0]]),
            "labels": torch.tensor([0, 1]),
            "scores": torch.tensor([0.9, 0.8]),
        }

    def test__id_maps__should_follow_label_order(
        self, detection_adapter: DetectionAdapter
    ) -> None:
        """Model ids are the positions in ``labels``, both ways."""
        assert detection_adapter.id2label == {0: "box", 1: "bag"}
        assert detection_adapter.label2id == {"box": 0, "bag": 1}

    def test__encode__should_give_model_category_ids_and_coco_boxes(
        self, detection_adapter: DetectionAdapter, box_image: ImageAnnotation
    ) -> None:
        """The target carries the image id and one COCO annotation per box."""
        target = detection_adapter.encode(box_image, self.IMAGE_ID)

        assert target["image_id"] == self.IMAGE_ID
        (annotation,) = target["annotations"]
        assert annotation["category_id"] == detection_adapter.label2id["box"]
        assert annotation["bbox"] == box_image.boxes[0].box.xywh

    def test__encode__when_no_box__should_give_no_annotation(
        self, detection_adapter: DetectionAdapter, empty_image: ImageAnnotation
    ) -> None:
        """An image without boxes is a negative example."""
        assert detection_adapter.encode(empty_image, 0)["annotations"] == []

    def test__decode__should_leave_degenerate_boxes_out(
        self, detection_adapter: DetectionAdapter, result: dict[str, torch.Tensor]
    ) -> None:
        """A zero-height box has no ``BoundingBox`` representation."""
        boxes = detection_adapter.decode(result)

        assert [box.label for box in boxes] == ["bag"]
        assert [box.score for box in boxes] == pytest.approx([0.8])
        assert boxes[0].box.xyxy == [1.0, 1.0, 10.0, 10.0]
