"""Detects boxes in one image at a time, whatever format the model is stored in."""

from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from baseline_core.core.ml.huggingface.detection.detection_adapter import DetectionAdapter
from baseline_core.core.ml.modality.image.annotation import BoundingBox, BoxAnnotation
from baseline_core.core.ml.model_runner import AbstractModelRunner


class Detector:
    """A detection model, its processor and its box adapter, applied to one image at a time.

    Attributes:
        runner: Runs the model, PyTorch or ONNX.
        processor: Hugging Face image processor preparing pixels and decoding raw outputs.
        adapter: Decodes the post-processed boxes.
        threshold: Minimum confidence of a returned box.
    """

    def __init__(
        self,
        runner: AbstractModelRunner,
        processor: Any,  # noqa: ANN401
        adapter: DetectionAdapter,
        threshold: float,
    ) -> None:
        """Keep the collaborators.

        Args:
            runner: Runs the model.
            processor: Prepares the pixels and decodes the raw outputs.
            adapter: Decodes the post-processed boxes.
            threshold: Minimum confidence of a returned box.
        """
        self.runner = runner
        self.processor = processor
        self.adapter = adapter
        self.threshold = threshold

    def __call__(
        self, image: Image.Image | np.ndarray, roi: BoundingBox | None = None
    ) -> list[BoxAnnotation]:
        """Detect the boxes of ``image``, or of its ``roi``, in the image's pixel coordinates.

        Args:
            image: RGB image, as PIL or an ``(h, w, 3)`` array.
            roi: Region to look at; boxes are shifted back into the full image.

        Returns:
            The scored boxes at or above ``threshold``.
        """
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)

        if roi is None:
            return self.detect(pil_image)

        crop_box = roi.clipped(pil_image.width, pil_image.height)
        left, top = int(crop_box.x1), int(crop_box.y1)
        crop = pil_image.crop((left, top, int(crop_box.x2), int(crop_box.y2)))
        return [box.shifted(left, top) for box in self.detect(crop)]

    def detect(self, image: Image.Image) -> list[BoxAnnotation]:
        """Run the model on the whole of ``image``.

        Args:
            image: RGB image.

        Returns:
            The scored boxes at or above ``threshold``.
        """
        inputs = dict(self.processor(images=[image], return_tensors="pt"))
        outputs = SimpleNamespace(**self.runner(inputs))
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(image.height, image.width)]
        )[0]
        return self.adapter.decode(result)
