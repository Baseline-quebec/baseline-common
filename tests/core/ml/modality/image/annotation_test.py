"""Tests for ``BoundingBox`` and ``BoxAnnotation``."""

import pytest

from baseline_core.core.ml.modality.image.annotation import BoundingBox, BoxAnnotation


class TestBoundingBox:
    """Tests for the geometry of ``BoundingBox``."""

    @pytest.fixture
    def box(self) -> BoundingBox:
        """A 10x20 box at ``(10, 10)``."""
        return BoundingBox(x1=10, y1=10, x2=20, y2=30)

    @pytest.mark.parametrize(("x2", "y2"), [(10, 30), (20, 10), (5, 30)])
    def test__init__when_corners_inverted_or_equal__should_raise_value_error(
        self, x2: float, y2: float
    ) -> None:
        """Empty and inverted boxes are rejected."""
        with pytest.raises(ValueError, match="x2 > x1 and y2 > y1"):
            BoundingBox(x1=10, y1=10, x2=x2, y2=y2)

    def test__from_xywh__should_round_trip_through_xywh(self, box: BoundingBox) -> None:
        """COCO ``[x, y, w, h]`` converts both ways."""
        rebuilt = BoundingBox.from_xywh(*box.xywh)

        assert rebuilt == box
        assert box.xywh == [10, 10, 10, 20]

    def test__area__should_multiply_width_and_height(self, box: BoundingBox) -> None:
        """Area of a 10x20 box."""
        assert (box.width, box.height, box.area) == (10, 20, 200)

    @pytest.mark.parametrize(
        ("other", "expected"),
        [
            (BoundingBox(x1=10, y1=10, x2=20, y2=30), 1.0),
            (BoundingBox(x1=15, y1=10, x2=25, y2=30), 100 / 300),
            (BoundingBox(x1=20, y1=10, x2=30, y2=30), 0.0),
        ],
    )
    def test__iou__should_divide_intersection_by_union(
        self, box: BoundingBox, other: BoundingBox, expected: float
    ) -> None:
        """Identical, half-overlapping and touching boxes."""
        assert box.iou(other) == pytest.approx(expected)

    def test__clipped__should_stay_inside_the_image(self) -> None:
        """Parts outside the image are cut off."""
        box = BoundingBox(x1=-5, y1=-5, x2=50, y2=50)

        clipped = box.clipped(width=40, height=30)

        assert clipped.xyxy == [0, 0, 40, 30]

    def test__clipped__when_outside_the_image__should_raise_value_error(
        self, box: BoundingBox
    ) -> None:
        """Nothing is left of a box beyond the image."""
        with pytest.raises(ValueError, match="outside the 5x5 image"):
            box.clipped(width=5, height=5)

    def test__shifted__should_translate_every_corner(self, box: BoundingBox) -> None:
        """Crop coordinates move back to the full image."""
        shifted = box.shifted(dx=100, dy=-5)

        assert shifted.xyxy == [110, 5, 120, 25]


class TestBoxAnnotation:
    """Tests for ``BoxAnnotation``."""

    def test__shifted__should_translate_the_box_and_keep_the_rest(self) -> None:
        """Label, score and occlusion travel with the box."""
        annotation = BoxAnnotation(
            label="bag", box=BoundingBox(x1=1, y1=2, x2=5, y2=6), score=0.5, occluded=True
        )

        shifted = annotation.shifted(10, 20)

        assert shifted == BoxAnnotation(
            label="bag", box=BoundingBox(x1=11, y1=22, x2=15, y2=26), score=0.5, occluded=True
        )
