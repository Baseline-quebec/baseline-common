"""Tests for ``BoxDrawer``."""

import pytest
from PIL import Image

from baseline_core.core.ml.modality.image.annotation import BoundingBox, BoxAnnotation
from baseline_core.core.ml.modality.image.box_drawer import BoxDrawer


class TestBoxDrawer:
    """Tests for drawing boxes on a copy of the image."""

    @pytest.fixture
    def image(self) -> Image.Image:
        """A black 160x120 image."""
        return Image.new("RGB", (160, 120))

    @pytest.fixture
    def drawer(self) -> BoxDrawer:
        """Drawer of two labels."""
        return BoxDrawer(["box", "bag"])

    @pytest.fixture
    def boxes(self) -> list[BoxAnnotation]:
        """One box of each label."""
        return [
            BoxAnnotation(label="box", box=BoundingBox(x1=10, y1=10, x2=80, y2=60)),
            BoxAnnotation(label="bag", box=BoundingBox(x1=90, y1=50, x2=150, y2=110)),
        ]

    def test__init__should_give_every_label_its_own_color(self, drawer: BoxDrawer) -> None:
        """Labels are told apart by color."""
        assert drawer.colors["box"] != drawer.colors["bag"]

    def test__draw__should_paint_a_copy_and_leave_the_image_untouched(
        self, drawer: BoxDrawer, image: Image.Image, boxes: list[BoxAnnotation]
    ) -> None:
        """The copy has the same size and painted pixels; the original stays black."""
        preview = drawer.draw(image, boxes)

        assert preview.size == image.size
        assert preview.getpixel((10, 30)) != (0, 0, 0)
        assert image.getpixel((10, 30)) == (0, 0, 0)

    def test__draw__when_label_unknown__should_raise_key_error(
        self, drawer: BoxDrawer, image: Image.Image
    ) -> None:
        """A label given no color is an error, not a silent default."""
        box = BoxAnnotation(label="crate", box=BoundingBox(x1=1, y1=1, x2=5, y2=5))

        with pytest.raises(KeyError, match="crate"):
            drawer.draw(image, [box])
