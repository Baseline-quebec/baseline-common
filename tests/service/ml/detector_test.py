"""Tests for ``Detector``, with the tiny RF-DETR run through PyTorch and through ONNX."""

from typing import ClassVar

import numpy as np
import pytest
from PIL import Image
from transformers import RfDetrImageProcessor

from baseline_core.core.ml.huggingface.detection.detection_adapter import DetectionAdapter
from baseline_core.core.ml.modality.image.annotation import BoundingBox, BoxAnnotation
from baseline_core.core.ml.model_runner import (
    AbstractModelRunner,
    OnnxModelRunner,
    TorchModelRunner,
)
from baseline_core.service.ml.detector import Detector


class TestDetector:
    """Tests for ``Detector``; every test runs on both runners."""

    ACCEPT_EVERYTHING: ClassVar[float] = 0.0

    @pytest.fixture(params=["torch", "onnx"])
    def runner(
        self,
        request: pytest.FixtureRequest,
        torch_runner: TorchModelRunner,
        onnx_runner: OnnxModelRunner,
    ) -> AbstractModelRunner:
        """The same model through PyTorch or through ONNX."""
        return torch_runner if request.param == "torch" else onnx_runner

    @pytest.fixture
    def detector(
        self,
        runner: AbstractModelRunner,
        rf_detr_processor: RfDetrImageProcessor,
        detection_adapter: DetectionAdapter,
    ) -> Detector:
        """Detector keeping every box, since the tiny model is untrained."""
        return Detector(
            runner, rf_detr_processor, detection_adapter, threshold=self.ACCEPT_EVERYTHING
        )

    @pytest.fixture
    def image(self) -> Image.Image:
        """A fixed random 128x96 RGB image, larger than the model input."""
        pixels = np.random.default_rng(1).integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        return Image.fromarray(pixels)

    def test__call__should_return_scored_boxes_in_image_pixels(
        self, detector: Detector, image: Image.Image
    ) -> None:
        """Boxes are scaled back from the model input to the image, labeled by name."""
        boxes = detector(image)

        assert boxes
        assert all(isinstance(box, BoxAnnotation) and box.score is not None for box in boxes)
        assert {box.label for box in boxes} <= set(detector.adapter.labels)
        assert all(box.box.x2 <= image.width and box.box.y2 <= image.height for box in boxes)

    def test__call__when_array__should_accept_it(
        self, detector: Detector, image: Image.Image
    ) -> None:
        """An ``(h, w, 3)`` array gives the same result as the PIL image."""
        assert detector(np.asarray(image)) == detector(image)

    def test__call__when_roi__should_shift_boxes_back(
        self, detector: Detector, image: Image.Image
    ) -> None:
        """Detections in a crop are reported in full-image coordinates."""
        roi = BoundingBox(x1=40, y1=20, x2=120, y2=90)
        crop = image.crop((40, 20, 120, 90))

        in_roi = detector(image, roi=roi)
        in_crop = detector(crop)

        assert in_roi == [box.shifted(40, 20) for box in in_crop]


class TestDetectorFormats:
    """The two formats give the same answer."""

    @pytest.fixture
    def noise_image(self) -> Image.Image:
        """A 100x72 image of random pixels."""
        return Image.fromarray(
            np.random.default_rng(2).integers(0, 256, size=(72, 100, 3), dtype=np.uint8)
        )

    @pytest.fixture
    def torch_detector(
        self,
        torch_runner: TorchModelRunner,
        rf_detr_processor: RfDetrImageProcessor,
        detection_adapter: DetectionAdapter,
    ) -> Detector:
        """The tiny RF-DETR through PyTorch, keeping every box."""
        return Detector(torch_runner, rf_detr_processor, detection_adapter, threshold=0.0)

    @pytest.fixture
    def onnx_detector(
        self,
        onnx_runner: OnnxModelRunner,
        rf_detr_processor: RfDetrImageProcessor,
        detection_adapter: DetectionAdapter,
    ) -> Detector:
        """The tiny RF-DETR through onnxruntime, keeping every box."""
        return Detector(onnx_runner, rf_detr_processor, detection_adapter, threshold=0.0)

    def test__call__should_not_depend_on_the_model_format(
        self, torch_detector: Detector, onnx_detector: Detector, noise_image: Image.Image
    ) -> None:
        """PyTorch and ONNX detections match box for box."""
        from_torch = torch_detector(noise_image)
        from_onnx = onnx_detector(noise_image)

        assert len(from_torch) == len(from_onnx) > 0
        assert [box.label for box in from_torch] == [box.label for box in from_onnx]

        for torch_box, onnx_box in zip(from_torch, from_onnx, strict=True):
            assert torch_box.box.xyxy == pytest.approx(onnx_box.box.xyxy, abs=1e-3)

        assert [box.score for box in from_torch] == pytest.approx(
            [box.score for box in from_onnx], abs=1e-5
        )
