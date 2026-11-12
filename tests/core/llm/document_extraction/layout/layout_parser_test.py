"""Unit tests for LayoutParser: lines, links and images read off a PDF."""

from pathlib import Path

from baseline_core.core.llm.document_extraction.layout.layout_parser import LayoutParser


class TestLayoutParser:
    @staticmethod
    def one_page_pdf(content: str) -> bytes:
        """Author a one-page Letter PDF painting one content stream in Helvetica."""
        objects = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            f"<< /Length {len(content.encode())} >>\nstream\n{content}\nendstream",
        ]
        out = "%PDF-1.4\n"
        offsets = []

        for number, body in enumerate(objects, start=1):
            offsets.append(len(out.encode()))
            out += f"{number} 0 obj\n{body}\nendobj\n"

        xref = len(out.encode())
        out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
        out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
        out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
        return out.encode()

    def test__a_space_painted_off_the_page_leaves_a_line_whole(self) -> None:
        """A space set far above the page, painted between a number and its title, moves no pen."""
        pdf = self.one_page_pdf(
            "BT /F1 12 Tf 79 600 Td (1.) Tj ET\n"
            "BT /F1 12 Tf 47 3000 Td ( ) Tj ET\n"
            "BT /F1 12 Tf 94 600 Td (Interpretation) Tj ET\n"
        )
        page = LayoutParser().parse(pdf).pages[0]
        assert [line.text for line in page.lines] == ["1. Interpretation"]

    def test__text_set_outside_the_page_box_is_not_read(self) -> None:
        """A line placed below the page's bottom edge is printed nowhere, so it is no line."""
        pdf = self.one_page_pdf(
            "BT /F1 12 Tf 79 600 Td (Sur la page) Tj ET\n"
            "BT /F1 16 Tf 79 -120 Td (Cartouche hors page) Tj ET\n"
        )
        page = LayoutParser().parse(pdf).pages[0]
        assert [line.text for line in page.lines] == ["Sur la page"]

    def test__every_link_naming_a_document_is_kept(self, link_kinds_pdf: bytes) -> None:
        """URI, GoToR and Launch links all surface the file they point at."""
        document = LayoutParser().parse(link_kinds_pdf)
        targets = {link.target for link in document.pages[0].links}
        assert targets == {"https://example.org/PA2025-245.pdf", "annexes/plan.pdf", "budget.xlsx"}

    def test__internal_navigation_is_dropped(self, link_kinds_pdf: bytes) -> None:
        """A GoTo link names no document, so only the three file-bearing links survive."""
        document = LayoutParser().parse(link_kinds_pdf)
        assert len(document.pages[0].links) == 3

    def test__extractable_text_is_told_page_by_page(self, scanned_page_pdf: bytes) -> None:
        """A text page reads True and a scanned one False, within the same document."""
        document = LayoutParser().parse(scanned_page_pdf)
        assert document.pages[0].text_is_extractable is True
        assert document.pages[1].text_is_extractable is False

    def test__painted_images_surface_with_their_extent(self, scanned_page_pdf: bytes) -> None:
        """The image page lists its placement; the text page lists none."""
        document = LayoutParser().parse(scanned_page_pdf)
        (image,) = document.pages[1].images
        assert image.kind == "raster"
        assert image.bbox.x1 > image.bbox.x0
        assert document.pages[0].images == []

    def test__a_vector_drawing_is_an_image_and_a_rule_is_not(self, fixtures_dir: Path) -> None:
        """A drawn box big both ways surfaces as vector; a thin line is a rule, left out."""
        document = LayoutParser().parse((fixtures_dir / "vector_and_rule.pdf").read_bytes())
        (image,) = document.pages[0].images
        assert image.kind == "vector"

    def test__a_dense_drawing_page_is_one_image_not_none(self, fixtures_dir: Path) -> None:
        """Above the cluster cap (2010 short rules) the page still surfaces one union image."""
        document = LayoutParser().parse((fixtures_dir / "dense_drawing.pdf").read_bytes())
        (image,) = document.pages[0].images
        assert image.kind == "vector"
        assert image.bbox.y1 - image.bbox.y0 >= 599
