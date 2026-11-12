"""Unit tests for the column reorder: a two-column page read left column first, then right."""

from pathlib import Path

import pytest

from baseline_core.core.llm.document_extraction.layout.column_reorder import (
    ColumnConfig,
    ColumnReorder,
)
from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutSpan,
    PageSize,
    TextBlock,
)
from baseline_core.core.llm.document_extraction.layout.layout_parser import (
    LayoutConfig,
    LayoutParser,
)


class TwoColumns:
    """Lines and pages set in two columns, French on the left and English on the right."""

    @staticmethod
    def line(text: str, x0: float, y0: float, x1: float) -> LayoutLine:
        """One 10-point line of text, set from x0 to x1 at the height y0."""
        return LayoutLine(
            text=text,
            bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y0 + 10.0),
            spans=[LayoutSpan(text=text, size=10.0, font="Helvetica")],
        )

    @staticmethod
    def page(*blocks: list[LayoutLine]) -> LayoutPage:
        """A Letter page carrying the blocks given, in that order."""
        return LayoutPage(
            number=1,
            size=PageSize(width=612.0, height=792.0),
            blocks=[TextBlock(lines=lines) for lines in blocks],
        )

    @staticmethod
    def left(index: int, y: float) -> LayoutLine:
        """A French line of the left column, justified from 72 to 296."""
        return TwoColumns.line(f"fr {index}", 72.0, y, 296.0)

    @staticmethod
    def right(index: int, y: float) -> LayoutLine:
        """An English line of the right column, justified from 320 to 544."""
        return TwoColumns.line(f"en {index}", 320.0, y, 544.0)

    @staticmethod
    def texts(laid: LayoutPage) -> list[str]:
        """The page's lines, as text, in the order it reads."""
        return [one.text for one in laid.lines]


class TestReorderPage:
    def test__a_two_column_page_reads_left_column_then_right(self) -> None:
        """Rows painted French-English in turn read every French line, then every English one."""
        rows = [
            (TwoColumns.left(i, 100.0 + 14 * i), TwoColumns.right(i, 100.0 + 14 * i))
            for i in range(8)
        ]
        painted = TwoColumns.page([one for row in rows for one in row])
        assert TwoColumns.texts(ColumnReorder().reorder_page(painted)) == [
            *(f"fr {i}" for i in range(8)),
            *(f"en {i}" for i in range(8)),
        ]

    def test__a_full_width_heading_keeps_its_place_between_zones(self) -> None:
        """A heading across the gutter closes the zone above: its columns are read before it."""
        above = [
            line_
            for i in range(6)
            for line_ in (TwoColumns.left(i, 100.0 + 14 * i), TwoColumns.right(i, 100.0 + 14 * i))
        ]
        heading = TwoColumns.line("SECTION 21-10-02 / SECTION 21-10-02", 72.0, 200.0, 544.0)
        below = [
            line_
            for i in range(6, 12)
            for line_ in (TwoColumns.left(i, 136.0 + 14 * i), TwoColumns.right(i, 136.0 + 14 * i))
        ]
        laid = ColumnReorder().reorder_page(TwoColumns.page([*above, heading, *below]))
        assert TwoColumns.texts(laid) == [
            *(f"fr {i}" for i in range(6)),
            *(f"en {i}" for i in range(6)),
            "SECTION 21-10-02 / SECTION 21-10-02",
            *(f"fr {i}" for i in range(6, 12)),
            *(f"en {i}" for i in range(6, 12)),
        ]

    def test__the_lines_set_on_a_heading_row_are_read_with_it(self) -> None:
        """A date either side of a page number centred in the gutter stays one header row."""
        header = [
            TwoColumns.line("Le 7 août 2023", 72.0, 40.0, 140.0),
            TwoColumns.line("Page 2", 300.0, 40.0, 316.0),
            TwoColumns.line("August 7, 2023", 476.0, 40.0, 544.0),
        ]
        rows = [
            (TwoColumns.left(i, 100.0 + 14 * i), TwoColumns.right(i, 100.0 + 14 * i))
            for i in range(8)
        ]
        laid = ColumnReorder().reorder_page(
            TwoColumns.page(header, [one for row in rows for one in row])
        )
        assert TwoColumns.texts(laid)[:3] == ["Le 7 août 2023", "Page 2", "August 7, 2023"]
        assert TwoColumns.texts(laid)[3:] == [
            *(f"fr {i}" for i in range(8)),
            *(f"en {i}" for i in range(8)),
        ]

    def test__a_one_column_page_is_returned_untouched(self) -> None:
        """Prose across the page, a short last line and a date set right: the very same page."""
        prose = [TwoColumns.line(f"ligne {i}", 72.0, 100.0 + 14 * i, 540.0) for i in range(20)]
        short = TwoColumns.line("fin du paragraphe.", 72.0, 380.0, 180.0)
        dated = TwoColumns.line("Signé le 14 mai 2020", 400.0, 420.0, 540.0)
        laid = TwoColumns.page(prose, [short], [dated])
        assert ColumnReorder().reorder_page(laid) is laid

    def test__a_table_whose_cells_straddle_the_gap_is_not_reordered(self) -> None:
        """Short cells crossing every candidate gutter leave too few lines one-sided to cut."""
        # Each row cuts its three cells at other places, so every candidate gutter runs
        # through most rows' cells: the cut a column needs is nowhere on the page.
        cells = [
            cell
            for row in range(15)
            for shift in [37.0 * (row % 5)]
            for cell in (
                TwoColumns.line(f"{row}.a", 72.0, 100.0 + 14 * row, 188.0 + shift),
                TwoColumns.line(f"{row}.b", 190.0 + shift, 100.0 + 14 * row, 330.0 + shift),
                TwoColumns.line(f"{row}.c", 332.0 + shift, 100.0 + 14 * row, 540.0),
            )
        ]
        laid = TwoColumns.page(cells)
        assert ColumnReorder().gutter(laid.lines, laid.size.width) is None
        assert ColumnReorder().reorder_page(laid) is laid

    def test__a_form_label_column_beside_its_values_is_not_reordered(self) -> None:
        """A narrow label column and a wide value column are a form, read row by row."""
        rows = [
            (
                TwoColumns.line(f"Libellé {i} :", 75.0, 100.0 + 14 * i, 180.0),
                TwoColumns.line(f"valeur {i}", 220.0, 100.0 + 14 * i, 540.0),
            )
            for i in range(8)
        ]
        laid = TwoColumns.page([one for row in rows for one in row])
        assert ColumnReorder().reorder_page(laid) is laid

    def test__a_table_whose_cell_gap_looks_like_a_gutter_keeps_its_rows(self) -> None:
        """A table with a clean cell gap mid-page is read row by row, never column by column."""
        # Five columns: item, request, reference | number, label, category.
        columns = [(80.0, 212.0), (232.0, 262.0), (291.0, 315.0), (348.0, 420.0), (438.0, 557.0)]
        cells = [
            TwoColumns.line(f"{row}.{index}", x0, 240.0 + 27 * row, x1)
            for row in range(10)
            for index, (x0, x1) in enumerate(columns)
        ]
        laid = TwoColumns.page(cells)
        assert ColumnReorder().gutter(laid.lines, laid.size.width) is None
        assert ColumnReorder().reorder_page(laid) is laid

    def test__a_bilingual_page_with_hanging_item_numbers_is_still_reordered(self) -> None:
        """Item numbers hung left of each language's titles are no table: the page is reordered."""
        rows = [
            one
            for i in range(8)
            for one in (
                TwoColumns.line(f"{i}.", 72.0, 100.0 + 14 * i, 88.0),
                TwoColumns.line(f"fr {i}", 106.0, 100.0 + 14 * i, 296.0),
                TwoColumns.line(f"{i}.", 320.0, 100.0 + 14 * i, 336.0),
                TwoColumns.line(f"en {i}", 354.0, 100.0 + 14 * i, 544.0),
            )
        ]
        laid = ColumnReorder().reorder_page(TwoColumns.page(rows))
        assert TwoColumns.texts(laid) == [
            *(text for i in range(8) for text in (f"{i}.", f"fr {i}")),
            *(text for i in range(8) for text in (f"{i}.", f"en {i}")),
        ]

    def test__a_page_painted_column_by_column_keeps_its_blocks(self) -> None:
        """Already in reading order, the page is returned as the file grouped it."""
        laid = TwoColumns.page(
            [TwoColumns.left(i, 100.0 + 14 * i) for i in range(8)],
            [TwoColumns.right(i, 100.0 + 14 * i) for i in range(8)],
        )
        assert ColumnReorder().reorder_page(laid) is laid

    def test__consecutive_lines_of_one_block_stay_one_block(self) -> None:
        """A paragraph the file painted as a block is cut where the other column interrupts it."""
        french = [TwoColumns.left(i, 100.0 + 14 * i) for i in range(8)]
        english = [TwoColumns.right(i, 100.0 + 14 * i) for i in range(8)]
        laid = ColumnReorder().reorder_page(
            TwoColumns.page(french[:4], [*english[:4], *french[4:]], english[4:])
        )
        assert [[one.text for one in block.lines] for block in laid.blocks] == [
            [f"fr {i}" for i in range(4)],
            [f"fr {i}" for i in range(4, 8)],
            [f"en {i}" for i in range(4)],
            [f"en {i}" for i in range(4, 8)],
        ]

    def test__the_furniture_verdict_survives_the_reorder(self) -> None:
        """A running header told on the page is still dropped from its text once reordered."""
        header = TwoColumns.line("Rapport annuel — revue du comité", 72.0, 30.0, 400.0)
        rows = [
            (TwoColumns.left(i, 100.0 + 14 * i), TwoColumns.right(i, 100.0 + 14 * i))
            for i in range(8)
        ]
        marked = TwoColumns.page(
            [header], [one for row in rows for one in row]
        ).with_header_footer(frozenset({header.furniture_key}))
        laid = ColumnReorder().reorder_page(marked)
        assert laid.text.splitlines() == [
            *(f"fr {i}" for i in range(8)),
            *(f"en {i}" for i in range(8)),
        ]


class TestReorder:
    @staticmethod
    def two_column() -> LayoutDocument:
        """A document of one two-column page, its rows painted French-English in turn."""
        rows = [
            (TwoColumns.left(i, 100.0 + 14 * i), TwoColumns.right(i, 100.0 + 14 * i))
            for i in range(8)
        ]
        return LayoutDocument(
            pages=[TwoColumns.page([one for row in rows for one in row])], outline=[]
        )

    def test__the_document_text_follows_the_new_order(self) -> None:
        """``document.text`` reads the French column whole before the English one."""
        text = ColumnReorder().reorder(self.two_column()).text
        assert text.splitlines() == [
            *(f"fr {i}" for i in range(8)),
            *(f"en {i}" for i in range(8)),
        ]

    def test__switched_off_the_document_is_returned_as_painted(self) -> None:
        """``enabled=False`` never looks: the very same document comes back."""
        document = self.two_column()
        assert ColumnReorder(ColumnConfig(enabled=False)).reorder(document) is document


class TestParse:
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

    def bilingual_pdf(self) -> bytes:
        """A page painting eight rows, each a French line at 72 pt then an English one at 340."""
        return self.one_page_pdf(
            "".join(
                f"BT /F1 10 Tf 72 {700 - 14 * i} Td (Ligne francaise {i}) Tj ET\n"
                f"BT /F1 10 Tf 340 {700 - 14 * i} Td (English line {i}) Tj ET\n"
                for i in range(8)
            )
        )

    def test__the_parse_reads_a_bilingual_page_column_by_column(self) -> None:
        """``from_bytes`` applies the reorder by default: the French column, then the English."""
        page_ = LayoutParser().parse(self.bilingual_pdf()).pages[0]
        assert page_.text.splitlines() == [
            *(f"Ligne francaise {i}" for i in range(8)),
            *(f"English line {i}" for i in range(8)),
        ]

    def test__the_switch_keeps_the_painting_order(self) -> None:
        """``ColumnConfig(enabled=False)`` parses the same page in the order the file paints it."""
        page_ = (
            LayoutParser(LayoutConfig(columns=ColumnConfig(enabled=False)))
            .parse(self.bilingual_pdf())
            .pages[0]
        )
        assert page_.text.splitlines()[:2] == ["Ligne francaise 0", "English line 0"]

    @pytest.mark.parametrize(
        "fixture",
        sorted((Path(__file__).parents[4] / "fixtures").glob("*.pdf")),
        ids=lambda path: path.name,
    )
    def test__a_one_column_fixture_parses_identically_either_way(self, fixture: Path) -> None:
        """No fixture PDF is two-column, so the reorder leaves every one of them byte for byte."""
        pdf = fixture.read_bytes()
        on = LayoutParser().parse(pdf)
        off = LayoutParser(LayoutConfig(columns=ColumnConfig(enabled=False))).parse(pdf)
        assert on.model_dump_json() == off.model_dump_json()
