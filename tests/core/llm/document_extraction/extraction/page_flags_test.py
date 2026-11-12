"""Tests of the page checks: a scan or a mostly raster page is image-heavy, a drawing is not."""

from baseline_core.core.llm.document_extraction.extraction.page_flags import PageFlags
from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    LayoutLine,
    LayoutPage,
    LayoutSpan,
    PageImage,
    PageSize,
    RasterImage,
    TextBlock,
    VectorDrawing,
)


class TestPageFlags:
    """Raster coverage decides, never a vector drawing."""

    def page(self, text: str, image: PageImage | None) -> LayoutPage:
        """A 100x100 page printing ``text`` (or nothing) beside one image (or none)."""
        lines = (
            [
                LayoutLine(
                    text=text,
                    bbox=BBox(x0=0, y0=0, x1=50, y1=10),
                    spans=[LayoutSpan(text=text, size=11.0, font="Helvetica")],
                )
            ]
            if text
            else []
        )
        return LayoutPage(
            number=1,
            size=PageSize(width=100.0, height=100.0),
            blocks=[TextBlock(lines=lines)] if lines else [],
            images=[image] if image else [],
        )

    def raster(self, side: float) -> RasterImage:
        """A square raster placement of ``side`` points."""
        return RasterImage(bbox=BBox(x0=0, y0=0, x1=side, y1=side))

    def test__a_scan_is_image_heavy(self) -> None:
        """No text and an image: the pixels are all there is."""
        assert PageFlags.is_image_heavy(self.page("", self.raster(10)), 0.3)

    def test__a_mostly_raster_page_is_image_heavy(self) -> None:
        """A photo over 60 % of a printed page carries what its text does not."""
        assert PageFlags.is_image_heavy(self.page("Figure 1", self.raster(80)), 0.3)

    def test__a_small_logo_is_not(self) -> None:
        """A text page with a small raster is read as text."""
        assert not PageFlags.is_image_heavy(self.page("Report", self.raster(20)), 0.3)

    def test__a_vector_drawing_is_never_image_heavy(self) -> None:
        """A drawn table is extractable text, already in the text reading."""
        drawing = VectorDrawing(bbox=BBox(x0=0, y0=0, x1=100, y1=100))

        assert not PageFlags.is_image_heavy(self.page("Table 2", drawing), 0.3)

    def test__an_empty_page_is_not(self) -> None:
        """Nothing printed, nothing painted: nothing to read."""
        assert not PageFlags.is_image_heavy(self.page("", None), 0.3)
