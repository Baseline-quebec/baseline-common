"""Adapter feeding ``ImageAnnotation`` to a Hugging Face detection image processor."""

from collections.abc import Callable, Mapping
from typing import Any

from baseline_core.core.ml.huggingface.detection.detection_adapter import DetectionAdapter
from baseline_core.core.ml.modality.image.annotation import ImageAnnotation


class DetectionProcessor:
    """A Hugging Face detection processor that takes ``ImageAnnotation`` as targets.

    Attributes:
        processor: The wrapped Hugging Face image processor.
        adapter: Encodes the annotations as the processor expects.
    """

    def __init__(
        self, processor: Callable[..., Mapping[str, Any]], adapter: DetectionAdapter
    ) -> None:
        """Keep the processor and the adapter.

        Args:
            processor: The Hugging Face image processor.
            adapter: Encodes the annotations as the processor expects.
        """
        self.processor = processor
        self.adapter = adapter

    def __call__(
        self,
        images: list[Any],
        annotations: list[ImageAnnotation],
        **kwargs: Any,  # noqa: ANN401
    ) -> Mapping[str, Any]:
        """Run the processor on ``images`` with ``annotations`` encoded as it expects.

        Args:
            images: One image per sample.
            annotations: The boxes of every image, in the same order.
            kwargs: Passed through to the processor (``return_tensors``, ...).

        Returns:
            The processor output: model inputs and ``labels``.
        """
        encoded = [
            self.adapter.encode(image, image_id) for image_id, image in enumerate(annotations)
        ]
        return self.processor(images=images, annotations=encoded, **kwargs)
