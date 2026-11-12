"""Tests for ``AnnotatedImageSource``."""

from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image
from torchvision import tv_tensors

from baseline_core.core.ml.modality.image.annotated_image_source import AnnotatedImageSource
from baseline_core.core.ml.modality.image.annotation import (
    BoundingBox,
    BoxAnnotation,
    ImageAnnotation,
)
from baseline_core.core.ml.modality.image.format.coco_codec import CocoCodec


class TestAnnotatedImageSource:
    """Tests for ``AnnotatedImageSource``."""

    IMAGE_COUNT: ClassVar[int] = 4

    @pytest.fixture
    def dataset_dir(self, tmp_path: Path) -> Path:
        """Four 16x8 images, each with a box and, on odd frames, a bag."""
        images = [
            ImageAnnotation(
                file_name=f"frame_{index}.png",
                width=16,
                height=8,
                boxes=[
                    BoxAnnotation(label="box", box=BoundingBox(x1=1, y1=1, x2=5, y2=5)),
                    *(
                        [BoxAnnotation(label="bag", box=BoundingBox(x1=6, y1=1, x2=9, y2=5))]
                        if index % 2
                        else []
                    ),
                ],
            )
            for index in reversed(range(self.IMAGE_COUNT))
        ]
        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "images").mkdir(parents=True)

        for image in images:
            Image.new("RGB", (16, 8)).save(dataset_dir / "images" / image.file_name)

        CocoCodec.write(dataset_dir / "annotations.json", images, ["box", "bag"])
        return dataset_dir

    @pytest.fixture
    def content_labels(self) -> list[str]:
        """The content model's labels."""
        return ["bag"]

    @pytest.fixture
    def source(
        self, dataset_dir: Path, content_labels: list[str], coco_codec: CocoCodec
    ) -> AnnotatedImageSource:
        """The content model's source over the folder."""
        return AnnotatedImageSource.from_annotations(
            dataset_dir / "annotations.json", dataset_dir / "images", content_labels, coco_codec
        )

    def test__from_annotations__should_index_every_image_sorted_by_file_name(
        self, source: AnnotatedImageSource
    ) -> None:
        """Images without a box of the model's labels stay, as negative examples."""
        file_names = [source.get_value(index, "annotations").file_name for index in range(4)]
        assert file_names == [f"frame_{index}.png" for index in range(self.IMAGE_COUNT)]
        assert source.field_names == ["image", "annotations"]

    def test__from_annotations__should_keep_only_the_boxes_of_the_model_labels(
        self, source: AnnotatedImageSource
    ) -> None:
        """The box model's boxes, in the same file, are left out."""
        labels = [
            [box.label for box in source.get_value(index, "annotations").boxes]
            for index in range(len(source))
        ]

        assert labels == [[], ["bag"], [], ["bag"]]

    def test__get_value__when_image__should_decode_the_file(
        self, source: AnnotatedImageSource
    ) -> None:
        """The record keeps the path; reading ``image`` decodes it."""
        image = source.get_value(0, "image")

        assert isinstance(source.get_stored_value(0, "image"), Path)
        assert isinstance(image, tv_tensors.Image)
        assert image.shape == (3, 8, 16)

    def test__from_annotations__when_image_file_missing__should_leave_it_out(
        self, dataset_dir: Path, content_labels: list[str], coco_codec: CocoCodec
    ) -> None:
        """A missing file is dropped when indexing, never read mid-epoch."""
        (dataset_dir / "images" / "frame_3.png").unlink()

        source = AnnotatedImageSource.from_annotations(
            dataset_dir / "annotations.json", dataset_dir / "images", content_labels, coco_codec
        )

        file_names = [
            source.get_value(index, "annotations").file_name for index in range(len(source))
        ]
        assert file_names == ["frame_0.png", "frame_1.png", "frame_2.png"]
