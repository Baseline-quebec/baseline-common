"""PageFlags — whether a page's pixels carry what its text does not, decided without a model."""

from baseline_core.core.llm.document_extraction.layout.layout import LayoutPage


class PageFlags:
    """Cheap page checks: each reads what the parse already holds, and costs nothing."""

    @staticmethod
    def raster_share(page: LayoutPage) -> float:
        """The share of the page's area its raster images cover.

        Args:
            page: The parsed page.

        Returns:
            Covered area over page area; 0 for a page without area.
        """
        area = page.size.width * page.size.height

        if area <= 0:
            return 0.0

        covered = sum(
            max(0.0, image.bbox.width) * max(0.0, image.bbox.height)
            for image in page.images
            if image.kind == "raster"
        )
        return min(1.0, covered / area)

    @classmethod
    def is_image_heavy(cls, page: LayoutPage, threshold: float) -> bool:
        """Whether the page carries what its text does not: a scan, or mostly raster.

        Args:
            page: The parsed page.
            threshold: Raster coverage above which pixels are read.

        Returns:
            True for a page with no text but an image, or raster above ``threshold``.
        """
        if not page.lines:
            return bool(page.images)

        return cls.raster_share(page) > threshold
