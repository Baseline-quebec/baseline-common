"""Cropping an image around a box at full resolution, and mapping boxes found there back."""

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.ml.modality.image.annotation import BoundingBox, BoxAnnotation


class Crop(BaseModel):
    """A crop of an image and where it sits in the image.

    Attributes:
        image: The crop, full-resolution pixels of the original.
        left: Horizontal offset of the crop in the original.
        top: Vertical offset of the crop in the original.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    left: int = Field(ge=0)
    top: int = Field(ge=0)

    def in_original(self, boxes: list[BoxAnnotation]) -> list[BoxAnnotation]:
        """Move boxes found in the crop back into the original's pixel coordinates.

        Args:
            boxes: Boxes in the crop's pixels.

        Returns:
            The same boxes in the original's pixels.
        """
        return [box.shifted(self.left, self.top) for box in boxes]


class BoxCropper:
    """Crops an image around a box, with a buffer on every side, without leaving the image."""

    @staticmethod
    def region(box: BoundingBox, width: int, height: int, margin: float) -> BoundingBox:
        """The box plus its margin, clipped to a ``width`` by ``height`` image.

        Args:
            box: Box to crop around.
            width: Image width.
            height: Image height.
            margin: Buffer added on every side, as a fraction of the box width or height.

        Returns:
            The region to crop.

        Raises:
            ValueError: If ``margin`` is negative.
        """
        if margin < 0:
            error_msg = f"Crop margin must be positive or zero, got {margin}."
            raise ValueError(error_msg)

        dx, dy = margin * box.width, margin * box.height
        return BoundingBox(x1=box.x1 - dx, y1=box.y1 - dy, x2=box.x2 + dx, y2=box.y2 + dy).clipped(
            width, height
        )

    @classmethod
    def crop(cls, image: Image.Image, box: BoundingBox, margin: float) -> Crop:
        """Cut the region around ``box`` out of ``image``.

        Args:
            image: Image to crop.
            box: Box to crop around, in the image's pixels.
            margin: Buffer added on every side of the box, as a fraction of its size.

        Returns:
            The crop and its offset in ``image``.
        """
        region = cls.region(box, image.width, image.height, margin)
        left, top = int(region.x1), int(region.y1)
        pixels = image.crop((left, top, int(region.x2), int(region.y2)))
        return Crop(image=pixels, left=left, top=top)
