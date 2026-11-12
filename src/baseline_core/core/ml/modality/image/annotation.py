"""Bounding boxes and the labeled boxes of one image."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """Axis-aligned box in pixels, corners ``(x1, y1)`` top-left and ``(x2, y2)`` bottom-right.

    Attributes:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge, greater than ``x1``.
        y2: Bottom edge, greater than ``y1``.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_corners(self) -> Self:
        """Reject empty or inverted boxes."""
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            error_msg = f"Box corners must satisfy x2 > x1 and y2 > y1, got {self.xyxy}."
            raise ValueError(error_msg)

        return self

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> "BoundingBox":
        """Build a box from its top-left corner and size, as COCO stores it.

        Args:
            x: Left edge.
            y: Top edge.
            width: Box width.
            height: Box height.

        Returns:
            The box.
        """
        return cls(x1=x, y1=y, x2=x + width, y2=y + height)

    @property
    def width(self) -> float:
        """Horizontal extent."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Vertical extent."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Width times height."""
        return self.width * self.height

    @property
    def xyxy(self) -> list[float]:
        """``[x1, y1, x2, y2]``."""
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def xywh(self) -> list[float]:
        """``[x1, y1, width, height]``, the COCO layout."""
        return [self.x1, self.y1, self.width, self.height]

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over union with ``other``.

        Args:
            other: Box to compare with.

        Returns:
            The ratio, between 0 and 1.
        """
        overlap_width = min(self.x2, other.x2) - max(self.x1, other.x1)
        overlap_height = min(self.y2, other.y2) - max(self.y1, other.y1)

        if overlap_width <= 0 or overlap_height <= 0:
            return 0.0

        intersection = overlap_width * overlap_height
        return intersection / (self.area + other.area - intersection)

    def clipped(self, width: float, height: float) -> "BoundingBox":
        """Clip the box to an image of ``width`` by ``height`` pixels.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            The part of the box inside the image.

        Raises:
            ValueError: If the box lies entirely outside the image.
        """
        if self.x1 >= width or self.y1 >= height or self.x2 <= 0 or self.y2 <= 0:
            error_msg = f"Box {self.xyxy} lies outside the {width}x{height} image."
            raise ValueError(error_msg)

        return BoundingBox(
            x1=max(self.x1, 0.0),
            y1=max(self.y1, 0.0),
            x2=min(self.x2, width),
            y2=min(self.y2, height),
        )

    def shifted(self, dx: float, dy: float) -> "BoundingBox":
        """Translate the box, e.g. from crop to full-image coordinates.

        Args:
            dx: Horizontal offset.
            dy: Vertical offset.

        Returns:
            The translated box.
        """
        return BoundingBox(x1=self.x1 + dx, y1=self.y1 + dy, x2=self.x2 + dx, y2=self.y2 + dy)


class BoxAnnotation(BaseModel):
    """One labeled box, annotated by hand or predicted.

    Attributes:
        label: Label name.
        box: Box in pixels.
        score: Confidence of a prediction; ``None`` on ground truth.
        occluded: Whether the object is partly hidden.
    """

    label: str
    box: BoundingBox
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    occluded: bool = False

    def shifted(self, dx: float, dy: float) -> "BoxAnnotation":
        """The same annotation with its box translated, e.g. from a crop to the full image.

        Args:
            dx: Horizontal offset.
            dy: Vertical offset.

        Returns:
            A copy with the translated box.
        """
        return self.model_copy(update={"box": self.box.shifted(dx, dy)})


class ImageAnnotation(BaseModel):
    """The boxes of one image.

    Attributes:
        file_name: Image path relative to the images folder.
        width: Image width in pixels.
        height: Image height in pixels.
        boxes: Labeled boxes.
    """

    file_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    boxes: list[BoxAnnotation] = Field(default_factory=list)
