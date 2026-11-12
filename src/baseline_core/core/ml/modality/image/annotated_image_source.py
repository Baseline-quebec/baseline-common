"""Annotated images as a data source: one record per image, its file and its boxes."""

from collections import Counter
from operator import attrgetter
from pathlib import Path
from typing import Any, Self

from loguru import logger

from baseline_core.core.ml.data.list_dict_source import ListDictSource
from baseline_core.core.ml.modality.image.annotation import ImageAnnotation
from baseline_core.core.ml.modality.image.format.abstract_annotation_codec import (
    AbstractAnnotationCodec,
)
from baseline_core.core.ml.modality.image.image_file_loader import ImageFileLoader


class AnnotatedImageSource(ListDictSource):
    """One record per annotated image: its ``image`` file, decoded when read, and annotations."""

    IMAGE_KEY = "image"
    ANNOTATIONS_KEY = "annotations"

    @classmethod
    def from_annotations(
        cls,
        annotations_path: Path,
        images_dir: Path,
        labels: list[str],
        codec: AbstractAnnotationCodec,
    ) -> Self:
        """Read the annotations file and index the images whose file exists.

        Args:
            annotations_path: Annotations file, in the format of ``codec``.
            images_dir: Folder holding the image files.
            labels: Labels the model learns; the boxes of other labels are left out.
            codec: Format of the annotations file.

        Returns:
            The source, records sorted by file name.
        """
        records = cls.index_records(codec.read(annotations_path), images_dir, labels)
        return cls(records, loaders={cls.IMAGE_KEY: ImageFileLoader()})

    @classmethod
    def index_records(
        cls, annotated_images: list[ImageAnnotation], images_dir: Path, labels: list[str]
    ) -> list[dict[str, Any]]:
        """Build one record per image whose file exists, sorted by file name, and log the rest.

        Args:
            annotated_images: Every image of the annotations file.
            images_dir: Folder holding the image files.
            labels: Labels the model learns; the boxes of other labels are left out.

        Returns:
            Records holding the image path under ``image`` and its ``annotations``.
        """
        records: list[dict[str, Any]] = []
        missing_files = []
        box_counts: Counter[str] = Counter()

        for image in sorted(annotated_images, key=attrgetter("file_name")):
            path = images_dir / image.file_name

            if not path.is_file():
                missing_files.append(image.file_name)
                continue

            annotations = cls.keep_boxes_of(image, labels)
            box_counts.update(box.label for box in annotations.boxes)
            records.append({cls.IMAGE_KEY: path, cls.ANNOTATIONS_KEY: annotations})

        if missing_files:
            logger.warning(
                "{} annotated images left out, file missing from {}: {}",
                len(missing_files),
                images_dir,
                missing_files,
            )

        logger.info(
            "{} images indexed; boxes per label: {}",
            len(records),
            {label: box_counts[label] for label in labels},
        )
        return records

    @staticmethod
    def keep_boxes_of(image: ImageAnnotation, labels: list[str]) -> ImageAnnotation:
        """Keep the boxes of ``image`` whose label is one of ``labels``.

        Args:
            image: Annotations of one image, possibly with boxes meant for other models.
            labels: Labels the model learns.

        Returns:
            A copy holding only the boxes of ``labels``.
        """
        kept = set(labels)
        return image.model_copy(
            update={"boxes": [box for box in image.boxes if box.label in kept]}
        )
