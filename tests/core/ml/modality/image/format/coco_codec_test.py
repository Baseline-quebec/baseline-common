"""Tests for ``CocoCodec``."""

from pathlib import Path
from typing import Any, ClassVar

import pytest

from baseline_core.core.ml.modality.image.annotation import (
    BoundingBox,
    BoxAnnotation,
    ImageAnnotation,
)
from baseline_core.core.ml.modality.image.format.coco_codec import CocoCodec


class TestCocoCodec:
    """Tests for the COCO encoding, decoding and file round trip."""

    PREDICTION_SCORE: ClassVar[float] = 0.8

    @pytest.fixture
    def cvat_document(self) -> dict[str, Any]:
        """A CVAT-style export: ids not starting at 1, categories out of order."""
        return {
            "categories": [
                {"id": 7, "name": "bag", "supercategory": ""},
                {"id": 3, "name": "box", "supercategory": ""},
            ],
            "images": [{"id": 42, "file_name": "a.png", "width": 64, "height": 48}],
            "annotations": [
                {
                    "id": 9,
                    "image_id": 42,
                    "category_id": 7,
                    "bbox": [1, 2, 10, 20],
                    "area": 200,
                    "iscrowd": 0,
                    "attributes": {"occluded": True},
                }
            ],
        }

    def test__decode__should_name_boxes_by_category(self, cvat_document: dict[str, Any]) -> None:
        """Each box takes the name of its category; boxes become pixel corners."""
        images = CocoCodec.decode(cvat_document)

        assert [image.file_name for image in images] == ["a.png"]
        assert images[0].boxes == [
            BoxAnnotation(label="bag", box=BoundingBox(x1=1, y1=2, x2=11, y2=22), occluded=True)
        ]

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [("category_id", 99, "unknown category ids: \\[99\\]"), ("image_id", 5, "image ids")],
    )
    def test__decode__when_unknown_reference__should_raise_value_error(
        self, cvat_document: dict[str, Any], field: str, value: int, message: str
    ) -> None:
        """Dangling categories and images are named in the error."""
        cvat_document["annotations"][0][field] = value

        with pytest.raises(ValueError, match=message):
            CocoCodec.decode(cvat_document)

    def test__encode__should_number_categories_and_images_from_one(
        self, annotated_images: list[ImageAnnotation], labels: list[str]
    ) -> None:
        """Category ids follow the label order; boxes are written as ``[x, y, w, h]``."""
        document = CocoCodec.encode(annotated_images, labels)

        assert [(c["id"], c["name"]) for c in document["categories"]] == [(1, "box"), (2, "bag")]
        assert [image["id"] for image in document["images"]] == [1, 2, 3]
        first = document["annotations"][0]
        assert (first["image_id"], first["category_id"], first["bbox"]) == (1, 1, [4, 4, 32, 24])
        assert (first["area"], first["iscrowd"]) == (768, 0)

    def test__encode__when_prediction__should_write_the_score(self) -> None:
        """Predictions keep their confidence."""
        box = BoxAnnotation(
            label="bag", box=BoundingBox(x1=0, y1=0, x2=5, y2=5), score=self.PREDICTION_SCORE
        )
        image = ImageAnnotation(file_name="a.png", width=10, height=10, boxes=[box])

        document = CocoCodec.encode([image], ["bag"])

        assert document["annotations"][0]["score"] == pytest.approx(self.PREDICTION_SCORE)

    def test__encode__when_box_label_not_a_category__should_raise_value_error(
        self, annotated_images: list[ImageAnnotation]
    ) -> None:
        """A box pointing to no category would make an invalid file."""
        with pytest.raises(ValueError, match=r"\['bag'\]"):
            CocoCodec.encode(annotated_images, ["box"])

    def test__write__then_read__should_round_trip(
        self, annotated_images: list[ImageAnnotation], labels: list[str], tmp_path: Path
    ) -> None:
        """A written file reads back to the same images."""
        path = tmp_path / "nested" / "annotations.json"

        CocoCodec.write(path, annotated_images, labels)

        assert CocoCodec.read(path) == annotated_images
