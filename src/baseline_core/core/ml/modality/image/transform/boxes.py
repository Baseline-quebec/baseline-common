"""The steps an image pipeline config may name: the box conversions and the ``v2`` steps."""

from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import (
    ColorJitter,
    RandomApply,
    RandomHorizontalFlip,
    RandomIoUCrop,
    RandomResizedCrop,
    Resize,
    SanitizeBoundingBoxes,
)

from baseline_core.core.ml.data.abstract_transform import AbstractTransform
from baseline_core.core.ml.modality.image.annotation import BoundingBox, ImageAnnotation


# A ``TransformPipeline`` resolves its step names in one module: the ``v2`` steps are exposed here.
__all__ = [
    "AnnotationsToBoundingBoxes",
    "BoundingBoxesToAnnotations",
    "ColorJitter",
    "RandomApply",
    "RandomHorizontalFlip",
    "RandomIoUCrop",
    "RandomResizedCrop",
    "Resize",
    "SanitizeBoundingBoxes",
]


class AnnotationsToBoundingBoxes(AbstractTransform):
    """Add the boxes of ``annotations`` as tensors that ``torchvision.transforms.v2`` moves."""

    BOXES_KEY: ClassVar[str] = "boxes"
    LABELS_KEY: ClassVar[str] = "labels"
    ANNOTATIONS_KEY: ClassVar[str] = "annotations"

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Add ``boxes`` and ``labels`` to ``data``.

        Args:
            data: Mapping with ``annotations``.

        Returns:
            ``data`` plus the box tensors.
        """
        annotations: ImageAnnotation = data[self.ANNOTATIONS_KEY]
        corners = torch.tensor(
            [box.box.xyxy for box in annotations.boxes], dtype=torch.float
        ).reshape(-1, 4)
        boxes = tv_tensors.BoundingBoxes(
            corners,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(annotations.height, annotations.width),
        )
        return {
            **data,
            self.BOXES_KEY: boxes,
            self.LABELS_KEY: torch.arange(len(annotations.boxes)),
        }


class BoundingBoxesToAnnotations(AbstractTransform):
    """Write the moved box tensors back into ``annotations``, the inverse of the step above."""

    IMAGE_KEY: ClassVar[str] = "image"

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Rebuild ``annotations`` from the box tensors.

        Args:
            data: Mapping with ``image``, ``annotations``, ``boxes`` and ``labels``.

        Returns:
            ``data`` with the updated ``annotations`` and no box tensors.

        Raises:
            ValueError: If a box collapsed to no area.
        """
        annotations: ImageAnnotation = data[AnnotationsToBoundingBoxes.ANNOTATIONS_KEY]
        corners = data[AnnotationsToBoundingBoxes.BOXES_KEY].tolist()
        indices = data[AnnotationsToBoundingBoxes.LABELS_KEY].tolist()
        collapsed = [
            corner for corner in corners if corner[2] <= corner[0] or corner[3] <= corner[1]
        ]

        if collapsed:
            error_msg = (
                f"Boxes of {annotations.file_name} collapsed to no area: {collapsed}; add "
                "SanitizeBoundingBoxes after the steps that can crop boxes away."
            )
            raise ValueError(error_msg)

        moved = [
            annotations.boxes[index].model_copy(
                update={"box": BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)}
            )
            for (x1, y1, x2, y2), index in zip(corners, indices, strict=True)
        ]
        height, width = data[self.IMAGE_KEY].shape[-2:]
        rest = {
            key: value
            for key, value in data.items()
            if key
            not in {AnnotationsToBoundingBoxes.BOXES_KEY, AnnotationsToBoundingBoxes.LABELS_KEY}
        }
        return {
            **rest,
            AnnotationsToBoundingBoxes.ANNOTATIONS_KEY: annotations.model_copy(
                update={"width": int(width), "height": int(height), "boxes": moved}
            ),
        }
