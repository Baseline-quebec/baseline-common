"""Tests for ``DetectionProcessor``."""

from typing import Any

import pytest
from PIL import Image
from transformers import RfDetrImageProcessor

from baseline_core.core.ml.huggingface.detection.detection_adapter import DetectionAdapter
from baseline_core.core.ml.huggingface.detection.detection_processor import DetectionProcessor
from baseline_core.core.ml.modality.image.annotation import ImageAnnotation


class RecordingProcessor:
    """Stands in for a Hugging Face processor: records its call and returns a fixed mapping."""

    def __init__(self) -> None:
        """Start with no recorded call."""
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record ``kwargs``."""
        self.kwargs = kwargs
        return {"pixel_values": "tensor", "labels": "targets"}


class TestDetectionProcessor:
    """Tests for ``DetectionProcessor``."""

    @pytest.fixture
    def images(
        self, box_image: ImageAnnotation, empty_image: ImageAnnotation
    ) -> list[Image.Image]:
        """Blank images of the annotated sizes."""
        return [
            Image.new("RGB", (image.width, image.height)) for image in (box_image, empty_image)
        ]

    def test__call__should_encode_annotations_for_the_wrapped_processor(
        self,
        detection_adapter: DetectionAdapter,
        images: list[Image.Image],
        box_image: ImageAnnotation,
        empty_image: ImageAnnotation,
    ) -> None:
        """Each image gets a COCO-style dict with model category ids; extras pass through."""
        wrapped = RecordingProcessor()
        processor = DetectionProcessor(wrapped, detection_adapter)

        result = processor(
            images=images, annotations=[box_image, empty_image], return_tensors="pt"
        )

        assert result == {"pixel_values": "tensor", "labels": "targets"}
        assert wrapped.kwargs["images"] == images
        assert wrapped.kwargs["return_tensors"] == "pt"
        first, second = wrapped.kwargs["annotations"]
        assert (first["image_id"], second["image_id"]) == (0, 1)
        assert second["annotations"] == []
        assert first["annotations"][0]["category_id"] == detection_adapter.label2id["box"]
        assert first["annotations"][0]["bbox"] == [4, 4, 32, 24]

    def test__call__should_feed_a_real_detection_processor(
        self,
        detection_adapter: DetectionAdapter,
        images: list[Image.Image],
        box_image: ImageAnnotation,
        empty_image: ImageAnnotation,
    ) -> None:
        """RF-DETR's processor accepts the encoding and returns one target per image."""
        hf_processor = RfDetrImageProcessor(size={"height": 64, "width": 64})
        processor = DetectionProcessor(hf_processor, detection_adapter)

        result = processor(
            images=images, annotations=[box_image, empty_image], return_tensors="pt"
        )

        assert set(result) >= {"pixel_values", "labels"}
        assert result["labels"][0]["class_labels"].tolist() == [detection_adapter.label2id["box"]]
        assert result["labels"][0]["orig_size"].tolist() == [box_image.height, box_image.width]
        assert result["labels"][1]["class_labels"].tolist() == []
