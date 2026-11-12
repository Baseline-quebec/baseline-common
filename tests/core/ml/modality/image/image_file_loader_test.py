"""Tests for ``ImageFileLoader``."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import tv_tensors

from baseline_core.core.ml.data.list_dict_source import ListDictSource
from baseline_core.core.ml.modality.image.image_file_loader import ImageFileLoader


class TestImageFileLoader:
    """Tests for ``ImageFileLoader``."""

    @pytest.fixture
    def gray_image_path(self, tmp_path: Path) -> Path:
        """A 40 x 20 grayscale PNG."""
        path = tmp_path / "gray.png"
        Image.new("L", (40, 20), color=128).save(path)
        return path

    @pytest.fixture
    def loader(self) -> ImageFileLoader:
        """Loader decoding image files."""
        return ImageFileLoader()

    @pytest.fixture
    def image_source(self, gray_image_path: Path, loader: ImageFileLoader) -> ListDictSource:
        """One record whose ``image`` path is decoded when read."""
        return ListDictSource(
            [{"image": gray_image_path, "label": "bag"}], loaders={"image": loader}
        )

    def test__load__should_decode_the_file_in_rgb_at_full_size(
        self, loader: ImageFileLoader, gray_image_path: Path
    ) -> None:
        """Three ``uint8`` channels whatever the mode on disk, height before width."""
        image = loader.load(gray_image_path)

        assert isinstance(image, tv_tensors.Image)
        assert (tuple(image.shape), image.dtype) == ((3, 20, 40), torch.uint8)

    def test__load__when_used_by_a_source__should_load_only_its_feature(
        self, image_source: ListDictSource
    ) -> None:
        """Records keep the path; the source decodes it when ``image`` is read."""
        assert isinstance(image_source.get_value(0, "image"), tv_tensors.Image)
        assert image_source.get_value(0, "label") == "bag"
