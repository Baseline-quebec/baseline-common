"""Unit tests for the layout tree's body accessors."""

from baseline_core.core.llm.document_extraction.layout.layout import LayoutPage


class TestLayoutPage:
    """The body accessors read images and links through the header/footer verdict."""

    def test__body_images_drop_the_header_logo(self, marked_page: LayoutPage) -> None:
        """The logo beside the running header goes; the drawing in the body stays."""
        (kept,) = marked_page.body_images
        assert kept.kind == "vector"

    def test__body_links_drop_the_footer_site_link(self, marked_page: LayoutPage) -> None:
        """The site link under the running footer goes; the annex link stays."""
        (kept,) = marked_page.body_links
        assert kept.target == "https://example.org/PA2025-1.pdf"

    def test__an_unmarked_page_keeps_everything(self, marked_page: LayoutPage) -> None:
        """Without a verdict nothing is furniture, exactly like ``body_lines``."""
        page = marked_page.with_header_footer(frozenset())
        assert page.body_images == page.images
        assert page.body_links == page.links
