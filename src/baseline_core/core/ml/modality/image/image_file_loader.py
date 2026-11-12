"""Loaders of the image formats, for data sources of image tasks."""

from pathlib import Path
from typing import override

from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image

from baseline_core.core.ml.data.abstract_feature_loader import AbstractFeatureLoader


class ImageFileLoader(AbstractFeatureLoader[Path | str, tv_tensors.Image]):
    """Decodes an image file as an RGB ``(3, height, width)`` ``uint8`` ``tv_tensors.Image``."""

    @override
    def load(self, stored: Path | str) -> tv_tensors.Image:
        """Decode the image file at ``stored``.

        Args:
            stored: Image file path.

        Returns:
            The image, in RGB whatever its mode on disk.
        """
        return tv_tensors.Image(decode_image(str(stored), mode=ImageReadMode.RGB))
