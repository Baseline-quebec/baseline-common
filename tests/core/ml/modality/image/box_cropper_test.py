"""Tests for ``BoxCropper`` and ``Crop``."""

import pytest
from PIL import Image

from baseline_core.core.ml.modality.image.annotation import BoundingBox, BoxAnnotation
from baseline_core.core.ml.modality.image.box_cropper import BoxCropper


class TestBoxCropper:
    """Tests for the region, the crop and the way back."""

    @pytest.fixture
    def image(self) -> Image.Image:
        """A 200x100 image."""
        return Image.new("RGB", (200, 100))

    @pytest.fixture
    def box(self) -> BoundingBox:
        """A 100x50 box at ``(50, 25)``."""
        return BoundingBox(x1=50, y1=25, x2=150, y2=75)

    def test__region__when_margin_negative__should_raise_value_error(
        self, box: BoundingBox
    ) -> None:
        """A negative margin is a configuration error."""
        with pytest.raises(ValueError, match="positive or zero"):
            BoxCropper.region(box, width=200, height=100, margin=-0.1)

    def test__region__should_add_the_margin_on_every_side(self, box: BoundingBox) -> None:
        """Ten percent of the width and of the height, on each side."""
        region = BoxCropper.region(box, width=200, height=100, margin=0.1)

        assert region.xyxy == [40, 20, 160, 80]

    def test__region__should_stay_inside_the_image(self, box: BoundingBox) -> None:
        """The margin is cut where it leaves the image."""
        region = BoxCropper.region(box, width=200, height=100, margin=1.0)

        assert region.xyxy == [0, 0, 200, 100]

    def test__crop__should_cut_the_region_and_remember_its_offset(
        self, image: Image.Image, box: BoundingBox
    ) -> None:
        """The crop is the region's pixels; the offset is the region's top-left corner."""
        crop = BoxCropper.crop(image, box, margin=0.1)

        assert crop.image.size == (120, 60)
        assert (crop.left, crop.top) == (40, 20)

    def test__in_original__should_shift_boxes_by_the_offset(
        self, image: Image.Image, box: BoundingBox
    ) -> None:
        """Boxes found in the crop land where the crop was."""
        crop = BoxCropper.crop(image, box, margin=0.0)
        found = BoxAnnotation(label="bag", box=BoundingBox(x1=10, y1=5, x2=30, y2=15))

        in_original = crop.in_original([found])

        assert in_original == [found.shifted(50, 25)]
