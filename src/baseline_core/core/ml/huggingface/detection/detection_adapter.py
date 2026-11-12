"""Conversions between annotated images and the boxes of Hugging Face detection processors."""

from typing import Any

import torch

from baseline_core.core.ml.modality.image.annotation import (
    BoundingBox,
    BoxAnnotation,
    ImageAnnotation,
)
from baseline_core.core.ml.modality.image.format.coco_codec import CocoCodec


class DetectionAdapter:
    """Encodes annotated images as detection targets and decodes predicted boxes.

    Attributes:
        labels: Label names in model id order.
    """

    def __init__(self, labels: list[str]) -> None:
        """Keep the labels.

        Args:
            labels: Label names in model id order.
        """
        self.labels = labels

    @property
    def label2id(self) -> dict[str, int]:
        """Model id of every label."""
        return {label: label_id for label_id, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        """Label of every model id."""
        return dict(enumerate(self.labels))

    def encode(self, image: ImageAnnotation, image_id: int) -> dict[str, Any]:
        """The COCO-style target of ``image`` that detection processors read.

        Args:
            image: Annotated image.
            image_id: Id of the image within its batch.

        Returns:
            ``image_id`` and the ``annotations`` of its boxes, with model category ids.
        """
        return {
            "image_id": image_id,
            "annotations": CocoCodec.encode_boxes(image, self.label2id, image_id),
        }

    def decode(self, result: dict[str, torch.Tensor]) -> list[BoxAnnotation]:
        """Turn one image's post-processed result into labeled, scored boxes.

        Args:
            result: Post-processed ``boxes`` (pixel ``xyxy``), ``labels`` and ``scores``.

        Returns:
            The boxes of non-zero width and height, in the result's order.
        """
        return [
            BoxAnnotation(
                label=self.id2label[label_id],
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                score=score,
            )
            for (x1, y1, x2, y2), label_id, score in zip(
                result["boxes"].detach().tolist(),
                result["labels"].tolist(),
                result["scores"].detach().tolist(),
                strict=True,
            )
            if x2 > x1 and y2 > y1
        ]
