"""Tests for the box conversions and the ``v2`` steps an image pipeline may name."""

from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from PIL import Image
from pytest_mock import MockerFixture
from torchvision import tv_tensors

from baseline_core.core.ml.data.base_dataset import (
    BaseDataset,
    BaseDatasetConfig,
    FeatureGroupConfig,
)
from baseline_core.core.ml.data.cache import DiskTransformCache
from baseline_core.core.ml.data.list_dict_source import ListDictSource
from baseline_core.core.ml.data.transform_pipeline import TransformConfig, TransformPipeline
from baseline_core.core.ml.modality.image.annotation import (
    BoundingBox,
    BoxAnnotation,
    ImageAnnotation,
)
from baseline_core.core.ml.modality.image.image_file_loader import ImageFileLoader
from baseline_core.core.ml.modality.image.transform import boxes
from baseline_core.core.ml.modality.image.transform.boxes import (
    AnnotationsToBoundingBoxes,
    BoundingBoxesToAnnotations,
)


class TestBoxConversions:
    """Tests for ``AnnotationsToBoundingBoxes`` and ``BoundingBoxesToAnnotations``."""

    @pytest.fixture
    def sample(self, box_image: ImageAnnotation) -> dict[str, Any]:
        """The 64x48 box image, blank, with its one occluded box."""
        return {"image": tv_tensors.Image(torch.zeros(3, 48, 64)), "annotations": box_image}

    @pytest.fixture
    def to_boxes(self) -> AnnotationsToBoundingBoxes:
        """Conversion of the annotations into box tensors."""
        return AnnotationsToBoundingBoxes()

    @pytest.fixture
    def to_annotations(self) -> BoundingBoxesToAnnotations:
        """Conversion of the box tensors back into annotations."""
        return BoundingBoxesToAnnotations()

    @pytest.fixture
    def flip_and_resize(self) -> TransformPipeline:
        """Conversions around a flip and a resize, named in configuration."""
        return TransformPipeline(
            module=boxes,
            transforms=[
                TransformConfig(name="AnnotationsToBoundingBoxes"),
                TransformConfig(name="RandomHorizontalFlip", params={"p": 1.0}),
                TransformConfig(name="Resize", params={"size": [24, 32]}),
                TransformConfig(name="BoundingBoxesToAnnotations"),
            ],
        )

    @pytest.fixture
    def augmented_dataset(self, sample: dict[str, Any]) -> BaseDataset:
        """``sample`` in a dataset whose one feature group flips image and boxes on training."""
        config = BaseDatasetConfig(
            feature_groups=[
                FeatureGroupConfig(
                    name="augmented",
                    source_keys=["image", "annotations"],
                    output_keys=["image", "annotations"],
                    pipeline=TransformPipeline(
                        module=boxes,
                        transforms=[
                            TransformConfig(name="AnnotationsToBoundingBoxes"),
                            TransformConfig(
                                name="RandomHorizontalFlip", params={"p": 1.0}, apply_on_eval=False
                            ),
                            TransformConfig(name="BoundingBoxesToAnnotations"),
                        ],
                    ),
                )
            ]
        )
        return BaseDataset(ListDictSource([sample]), config)

    def test__round_trip__without_steps__should_give_back_the_annotations(
        self,
        to_boxes: AnnotationsToBoundingBoxes,
        to_annotations: BoundingBoxesToAnnotations,
        sample: dict[str, Any],
        box_image: ImageAnnotation,
    ) -> None:
        """Converting there and back changes nothing and removes the box tensors."""
        result = to_annotations(to_boxes(sample))

        assert result.keys() == {"image", "annotations"}
        assert result["annotations"] == box_image

    def test__pipeline__when_flipped_and_resized__should_move_the_boxes_with_the_pixels(
        self, flip_and_resize: TransformPipeline, sample: dict[str, Any]
    ) -> None:
        """Steps named in configuration move the boxes; label and occlusion are kept."""
        result = flip_and_resize(sample)

        annotations = result["annotations"]
        assert (annotations.width, annotations.height) == (32, 24)
        assert annotations.boxes == [
            BoxAnnotation(label="box", box=BoundingBox(x1=14, y1=2, x2=30, y2=14), occluded=True)
        ]

    def test__call__when_a_box_collapsed__should_raise_value_error(
        self,
        to_boxes: AnnotationsToBoundingBoxes,
        to_annotations: BoundingBoxesToAnnotations,
        sample: dict[str, Any],
    ) -> None:
        """A box cropped to nothing names the fix instead of failing deep in validation."""
        converted = to_boxes(sample)
        converted["boxes"] = tv_tensors.BoundingBoxes(
            [[10, 10, 10, 20]], format="XYXY", canvas_size=(48, 64)
        )

        with pytest.raises(ValueError, match="SanitizeBoundingBoxes"):
            to_annotations(converted)

    def test__base_dataset__when_group_writes_image_and_annotations__should_augment_both(
        self, augmented_dataset: BaseDataset
    ) -> None:
        """One feature group reads and writes both keys; evaluation views skip the flip."""
        flipped = augmented_dataset[0]["annotations"].boxes[0].box
        unflipped = augmented_dataset.select([0], training=False)[0]["annotations"].boxes[0].box

        assert (flipped.x1, flipped.x2) == (28, 60)
        assert (unflipped.x1, unflipped.x2) == (4, 36)


class TestCachedResize:
    """Tests for a ``Resize`` cached on disk before a live flip, on a real image file."""

    EXPECTED_SIZE: ClassVar[tuple[int, int]] = (32, 24)
    EXPECTED_RESIZED_BOX: ClassVar[BoundingBox] = BoundingBox(x1=2, y1=2, x2=18, y2=14)
    EXPECTED_FLIPPED_BOX: ClassVar[BoundingBox] = BoundingBox(x1=14, y1=2, x2=30, y2=14)

    @pytest.fixture
    def image_path(self, tmp_path: Path, box_image: ImageAnnotation) -> Path:
        """A blank PNG of the box image's size."""
        path = tmp_path / box_image.file_name
        Image.new("RGB", (box_image.width, box_image.height)).save(path)
        return path

    @pytest.fixture
    def source(self, image_path: Path, box_image: ImageAnnotation) -> ListDictSource:
        """One record: the image file, decoded when read, and its one occluded box."""
        return ListDictSource(
            [{"image": image_path, "annotations": box_image}], loaders={"image": ImageFileLoader()}
        )

    @pytest.fixture
    def resized_config(self) -> BaseDatasetConfig:
        """Image and boxes resized to 32x24 once, then flipped on every training read."""
        return BaseDatasetConfig(
            feature_groups=[
                FeatureGroupConfig(
                    name="augmented",
                    source_keys=["image", "annotations"],
                    output_keys=["image", "annotations"],
                    pipeline=TransformPipeline(
                        module=boxes,
                        transforms=[
                            TransformConfig(
                                name="AnnotationsToBoundingBoxes", apply_on_the_fly=False
                            ),
                            TransformConfig(
                                name="Resize", params={"size": [24, 32]}, apply_on_the_fly=False
                            ),
                            TransformConfig(
                                name="BoundingBoxesToAnnotations", apply_on_the_fly=False
                            ),
                            TransformConfig(name="AnnotationsToBoundingBoxes"),
                            TransformConfig(
                                name="RandomHorizontalFlip", params={"p": 1.0}, apply_on_eval=False
                            ),
                            TransformConfig(name="BoundingBoxesToAnnotations"),
                        ],
                    ),
                )
            ]
        )

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Folder of the cached results."""
        return tmp_path / "cache"

    @pytest.fixture
    def dataset(
        self, source: ListDictSource, resized_config: BaseDatasetConfig, cache_dir: Path
    ) -> BaseDataset:
        """Training dataset over the image, with its resize cached in ``cache_dir``."""
        return BaseDataset(source, resized_config, cache=DiskTransformCache(cache_dir))

    def test__getitem__should_resize_the_image_and_move_its_boxes(
        self, dataset: BaseDataset
    ) -> None:
        """Pixels and boxes are resized by the cached steps, then flipped by the live ones."""
        sample = dataset[0]

        width, height = self.EXPECTED_SIZE
        assert tuple(sample["image"].shape) == (3, height, width)
        annotations = sample["annotations"]
        assert (annotations.width, annotations.height) == self.EXPECTED_SIZE
        assert annotations.boxes == [
            BoxAnnotation(label="box", box=self.EXPECTED_FLIPPED_BOX, occluded=True)
        ]

    def test__getitem__when_read_again__should_decode_the_image_once(
        self, dataset: BaseDataset, cache_dir: Path, mocker: MockerFixture
    ) -> None:
        """The second read loads the resized sample from disk and only flips it."""
        load = mocker.spy(ImageFileLoader, "load")
        first = dataset[0]

        second = dataset[0]

        assert load.call_count == 1
        assert len(list(cache_dir.glob("*.pt"))) == 1
        assert torch.equal(second["image"], first["image"])
        assert second["annotations"] == first["annotations"]

    def test__select__when_not_training__should_resize_without_flipping(
        self, dataset: BaseDataset
    ) -> None:
        """The evaluation split reads the same cached resize and skips the flip."""
        dataset[0]

        annotations = dataset.select([0], training=False)[0]["annotations"]

        assert (annotations.width, annotations.height) == self.EXPECTED_SIZE
        assert [box.box for box in annotations.boxes] == [self.EXPECTED_RESIZED_BOX]
