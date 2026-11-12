"""The COCO JSON format, as CVAT exports and imports it."""

import json
from pathlib import Path
from typing import Any, ClassVar

from baseline_core.core.ml.modality.image.annotation import (
    BoundingBox,
    BoxAnnotation,
    ImageAnnotation,
)
from baseline_core.core.ml.modality.image.format.abstract_annotation_codec import (
    AbstractAnnotationCodec,
)


class CocoCodec(AbstractAnnotationCodec):
    """Reads and writes one COCO JSON file."""

    SUFFIX: ClassVar[str] = ".json"

    @staticmethod
    def read(path: Path) -> list[ImageAnnotation]:
        """Read a COCO file.

        Args:
            path: JSON file.

        Returns:
            The images, in file order.
        """
        return CocoCodec.decode(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def write(path: Path, images: list[ImageAnnotation], labels: list[str]) -> None:
        """Write a COCO file; parent folders are created.

        Args:
            path: JSON file.
            images: Annotated images.
            labels: The file's categories; every box label must be one of them.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(CocoCodec.encode(images, labels), indent=2), encoding="utf-8")

    @staticmethod
    def encode(images: list[ImageAnnotation], labels: list[str]) -> dict[str, Any]:
        """Build the COCO document of ``images``.

        Args:
            images: Annotated images.
            labels: The document's categories, numbered from 1 in this order.

        Returns:
            The document, ready for ``json.dumps``.

        Raises:
            ValueError: Listing the box labels missing from ``labels``.
        """
        undeclared = sorted({box.label for image in images for box in image.boxes} - set(labels))

        if undeclared:
            error_msg = f"Boxes use labels missing from the categories {labels}: {undeclared}"
            raise ValueError(error_msg)

        category_ids = {label: index for index, label in enumerate(labels, 1)}
        coco_images: list[dict[str, Any]] = []
        coco_annotations: list[dict[str, Any]] = []

        for image_id, image in enumerate(images, 1):
            coco_images.append({
                "id": image_id,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
            })

            for annotation in CocoCodec.encode_boxes(image, category_ids, image_id):
                coco_annotations.append({"id": len(coco_annotations) + 1, **annotation})

        categories = [
            {"id": category_id, "name": label, "supercategory": ""}
            for label, category_id in category_ids.items()
        ]
        return {"images": coco_images, "annotations": coco_annotations, "categories": categories}

    @staticmethod
    def encode_boxes(
        image: ImageAnnotation, category_ids: dict[str, int], image_id: int
    ) -> list[dict[str, Any]]:
        """Build the COCO annotations of one image's boxes, without annotation ids.

        Args:
            image: Annotated image.
            category_ids: Category id of every label.
            image_id: Id of the image the annotations point to.

        Returns:
            One annotation per box.
        """
        annotations = []

        for box in image.boxes:
            annotation: dict[str, Any] = {
                "image_id": image_id,
                "category_id": category_ids[box.label],
                "bbox": box.box.xywh,
                "area": box.box.area,
                "iscrowd": 0,
                "attributes": {"occluded": box.occluded},
            }

            if box.score is not None:
                annotation["score"] = box.score

            annotations.append(annotation)

        return annotations

    @staticmethod
    def decode(document: dict[str, Any]) -> list[ImageAnnotation]:
        """Read the annotated images of a COCO document; each box takes its category's name.

        Args:
            document: Decoded COCO JSON.

        Returns:
            The images, in document order.

        Raises:
            ValueError: If annotations reference unknown categories or images.
        """
        label_by_id = {category["id"]: category["name"] for category in document["categories"]}
        image_by_id = {
            image["id"]: ImageAnnotation(
                file_name=image["file_name"], width=image["width"], height=image["height"]
            )
            for image in document["images"]
        }
        annotations = document["annotations"]
        unknown_categories = sorted({
            annotation["category_id"]
            for annotation in annotations
            if annotation["category_id"] not in label_by_id
        })

        if unknown_categories:
            error_msg = f"Annotations reference unknown category ids: {unknown_categories}"
            raise ValueError(error_msg)

        unknown_images = sorted({
            annotation["image_id"]
            for annotation in annotations
            if annotation["image_id"] not in image_by_id
        })

        if unknown_images:
            error_msg = f"Annotations reference unknown image ids: {unknown_images}"
            raise ValueError(error_msg)

        for annotation in annotations:
            image_by_id[annotation["image_id"]].boxes.append(
                BoxAnnotation(
                    label=label_by_id[annotation["category_id"]],
                    box=BoundingBox.from_xywh(*annotation["bbox"]),
                    score=annotation.get("score"),
                    occluded=annotation.get("attributes", {}).get("occluded", False),
                )
            )

        return list(image_by_id.values())
