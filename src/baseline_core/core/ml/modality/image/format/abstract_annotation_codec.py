"""Contract of a codec reading and writing annotation files."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from baseline_core.core.ml.modality.image.annotation import ImageAnnotation


class AbstractAnnotationCodec(ABC):
    """Reads and writes annotated images in one file format (COCO, YOLO, Pascal VOC, ...).

    Attributes:
        SUFFIX: Ending of the paths this format is stored at, e.g. ``".json"``; empty for a folder.
    """

    SUFFIX: ClassVar[str]

    @staticmethod
    @abstractmethod
    def read(path: Path) -> list[ImageAnnotation]:
        """Read the annotated images stored at ``path``.

        Args:
            path: File or folder, depending on the format.

        Returns:
            The annotated images.
        """

    @staticmethod
    @abstractmethod
    def write(path: Path, images: list[ImageAnnotation], labels: list[str]) -> None:
        """Store ``images`` at ``path``.

        Args:
            path: File or folder, depending on the format.
            images: Annotated images.
            labels: Labels the file declares; every box label must be one of them.
        """
