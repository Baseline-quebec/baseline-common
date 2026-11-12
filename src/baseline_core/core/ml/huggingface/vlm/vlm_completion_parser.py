"""Reading the boxes a vision-language model completes, from its 0 to 1000 grid to pixels."""

import json
import re
from typing import ClassVar

from baseline_core.core.ml.modality.image.annotation import BoundingBox, BoxAnnotation


class VlmCompletionParser:
    """Turns a completion into labeled boxes in the pixels of the image sent.

    Attributes:
        GRID: Size of the coordinate grid of the completion, along both axes.
        LINE_PATTERN: ``label x1 y1 x2 y2`` lines, the format the prompt asks for.
        JSON_PATTERN: ``{"bbox_2d": [...], "label": ...}`` objects, which Qwen models emit anyway.
    """

    GRID: ClassVar[int] = 1000
    LINE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*([a-z_]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.MULTILINE
    )
    JSON_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\{[^{}]*\}")

    def parse(
        self, completion: str, labels: list[str], width: int, height: int
    ) -> list[BoxAnnotation]:
        """Read the boxes of ``completion``; repeats, other labels and empty boxes are dropped.

        Args:
            completion: The model's completion.
            labels: Labels kept, the ones the prompt asked for.
            width: Width of the image sent, in pixels.
            height: Height of the image sent, in pixels.

        Returns:
            The boxes, in completion order, without scores.
        """
        kept = set(labels)
        boxes: list[BoxAnnotation] = []
        seen: set[tuple[str, tuple[float, ...]]] = set()

        for label, coordinates in self.candidates(completion):
            key = (label, tuple(coordinates))

            if label not in kept or key in seen:
                continue

            seen.add(key)
            x1, y1, x2, y2 = (min(max(value, 0.0), self.GRID) / self.GRID for value in coordinates)
            corners = (x1 * width, y1 * height, x2 * width, y2 * height)

            if corners[2] > corners[0] and corners[3] > corners[1]:
                boxes.append(
                    BoxAnnotation(
                        label=label,
                        box=BoundingBox(
                            x1=corners[0], y1=corners[1], x2=corners[2], y2=corners[3]
                        ),
                    )
                )

        return boxes

    @classmethod
    def candidates(cls, completion: str) -> list[tuple[str, list[float]]]:
        """Every ``(label, [x1, y1, x2, y2])`` of ``completion``, in either format.

        Args:
            completion: The model's completion.

        Returns:
            The lines first, then the JSON objects, as written; nothing is checked yet.
        """
        found = [
            (label, [float(value) for value in values])
            for label, *values in cls.LINE_PATTERN.findall(completion)
        ]

        for match in cls.JSON_PATTERN.finditer(completion):
            try:
                item = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

            if not isinstance(item, dict) or "label" not in item:
                continue

            coordinates = item.get("bbox_2d")

            if isinstance(coordinates, list) and len(coordinates) == 4:  # noqa: PLR2004
                found.append((str(item["label"]), [float(value) for value in coordinates]))

        return found
