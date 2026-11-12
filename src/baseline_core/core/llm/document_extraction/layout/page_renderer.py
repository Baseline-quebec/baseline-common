"""PageRenderer — the one door to the library for pixels: pages and bands as PNG."""

from io import BytesIO

import pypdfium2 as pdfium

from baseline_core.core.llm.document_extraction.layout.layout import BBox, PageRegion, PageSize
from baseline_core.core.llm.document_extraction.layout.layout_parser import LayoutParser


class PageRenderer:
    """Renders pages and bands of a document as PNG images, for a model that reads pixels."""

    @classmethod
    def render_regions(cls, pdf: bytes, regions: list[PageRegion], *, dpi: int) -> list[bytes]:
        """Render bands of pages.

        Args:
            pdf: Raw bytes of the document.
            regions: The bands to cut, in the order they should render.
            dpi: Render resolution; the caller's agent carries the measured value.

        Returns:
            One PNG per requested region, in the same order.
        """
        with LayoutParser.PDFIUM_LOCK:
            doc = pdfium.PdfDocument(pdf)

            try:
                return [
                    cls._png(doc[region.page - 1], dpi=dpi, clip=region.clip) for region in regions
                ]
            finally:
                doc.close()

    @staticmethod
    def _png(page: pdfium.PdfPage, *, dpi: int, clip: BBox | None = None) -> bytes:
        """Render one page, whole or clipped, to PNG bytes.

        Args:
            page: The library's page object.
            dpi: Render resolution.
            clip: Extent to keep, in top-down points; None renders the whole page.

        Returns:
            The PNG bytes.
        """
        crop = (0.0, 0.0, 0.0, 0.0)

        if clip is not None:
            width, height = page.get_size()
            # The library trims from each edge, measured bottom-up; the clip is top-down.
            crop = (clip.x0, height - clip.y1, width - clip.x1, clip.y0)

        image = page.render(scale=dpi / PageSize.POINTS_PER_INCH, crop=crop).to_pil()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
