"""Drawing labeled boxes on an image, to look at annotations or predictions."""

from typing import ClassVar

from PIL import Image, ImageDraw, ImageFont

from baseline_core.core.ml.modality.image.annotation import BoxAnnotation


class BoxDrawer:
    """Draws labeled boxes on a copy of an image, one color per label.

    Attributes:
        PALETTE: Colors handed out to the labels, in order, cycling past the end.
        colors: Color of every label.
    """

    PALETTE: ClassVar[tuple[str, ...]] = (
        "#e6194b",
        "#3cb44b",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#008080",
        "#9a6324",
        "#f032e6",
        "#469990",
        "#808000",
    )

    def __init__(self, labels: list[str]) -> None:
        """Assign a color to every label.

        Args:
            labels: Labels the boxes may carry.
        """
        self.colors = {
            label: self.PALETTE[index % len(self.PALETTE)] for index, label in enumerate(labels)
        }

    def draw(self, image: Image.Image, boxes: list[BoxAnnotation]) -> Image.Image:
        """Draw ``boxes`` with their labels on a copy of ``image``.

        Args:
            image: Image the boxes belong to.
            boxes: Boxes in the image's pixels.

        Returns:
            The annotated copy, ``image`` untouched; a label given no color is a ``KeyError``.
        """
        canvas = image.copy()
        pen = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=max(16, image.width // 50))
        line_width = max(3, image.width // 300)

        for box in boxes:
            color = self.colors[box.label]
            pen.rectangle(box.box.xyxy, outline=color, width=line_width)
            pen.text(
                (box.box.x1 + 4, box.box.y1 + 2),
                box.label,
                fill=color,
                font=font,
                stroke_width=3,
                stroke_fill="white",
            )

        return canvas
