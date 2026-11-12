"""Unit tests for the sheet heading finder: a heading is set above its own sheet's body, alone."""

from baseline_core.core.llm.document_extraction.indexing.detection.sheet_heading_finder import (
    SheetHeadingFinder,
)
from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutSpan,
    OutlineEntry,
    PageSize,
    TextBlock,
)
from baseline_core.core.llm.document_extraction.sheet_parser import SheetParser


class TestScan:
    @staticmethod
    def row(cells: list[tuple[str, float, bool]], index: int, column: int = 0) -> LayoutLine:
        """Build one row of cells at a row index.

        Args:
            cells: Each cell as (text, size, bold).
            index: The row index.
            column: The column the row opens on.

        Returns:
            The row, as the sheet parser lays it out.
        """
        spans = [
            LayoutSpan(text=text, size=size, font="Calibri", flags=LayoutSpan.BOLD if bold else 0)
            for text, size, bold in cells
        ]
        return LayoutLine(
            text=" ".join(span.text for span in spans),
            bbox=BBox(
                x0=48.0 * column,
                y0=15.0 * index,
                x1=48.0 * (column + len(cells)),
                y1=15.0 * (index + 1),
            ),
            spans=spans,
        )

    @staticmethod
    def sheet(number: int, rows: list[LayoutLine]) -> LayoutPage:
        """Build one sheet holding the rows as one block.

        Args:
            number: The sheet's 1-based page.
            rows: Its rows.

        Returns:
            The sheet.
        """
        return LayoutPage(
            number=number,
            size=PageSize(width=200.0, height=15.0 * (len(rows) + 1)),
            blocks=[TextBlock(lines=rows)],
        )

    def test__the_real_workbook_yields_its_title_and_its_subsection(
        self, workbook_bytes: bytes
    ) -> None:
        """A 14pt title and a lone bold row are headings; a bold header band and total are not."""
        found = SheetHeadingFinder.scan(SheetParser.parse(workbook_bytes))

        assert [candidate.text for candidate in found.format] == [
            "Situation financière",
            "Actifs financiers",
            "Nomenclature",
        ]

    def test__sheet_names_become_outline_candidates(self, workbook_bytes: bytes) -> None:
        """Each kept sheet's name reaches the reconciler as outline metadata on its page."""
        found = SheetHeadingFinder.scan(SheetParser.parse(workbook_bytes))

        assert [(c.text, c.page) for c in found.outline] == [("Bilan", 1), ("Notes", 3)]

    def test__each_sheet_is_judged_against_its_own_body(self) -> None:
        """A 10pt row heads a 9pt sheet even when the workbook's other sheet is set at 11pt."""
        small = self.sheet(
            1,
            [
                self.row([("Title", 10.0, False)], 0),
                self.row([("a", 9.0, False)], 1),
                self.row([("b", 9.0, False)], 2),
            ],
        )
        large = self.sheet(
            2, [self.row([("Other", 11.0, False)], 0), self.row([("c", 11.0, False)], 1)]
        )
        document = LayoutDocument(origin="cells", pages=[small, large], outline=[])

        found = SheetHeadingFinder.scan(document)

        assert [(c.text, c.page, c.position) for c in found.format] == [("Title", 1, 0.0)]

    def test__a_row_off_the_opening_column_is_not_a_heading(self) -> None:
        """A bold cell indented from the sheet's first column is a label, not a heading."""
        page = self.sheet(
            1,
            [
                self.row([("Title", 11.0, True)], 0, column=1),
                self.row([("a", 11.0, False)], 1),
                self.row([("b", 11.0, False)], 2),
            ],
        )

        found = SheetHeadingFinder.scan(LayoutDocument(origin="cells", pages=[page], outline=[]))

        assert found.format == []

    def test__a_mostly_bold_sheet_states_nothing_by_being_bold(self) -> None:
        """When the body itself is bold, a lone bold row at body size is not a heading."""
        page = self.sheet(
            1,
            [
                self.row([("Title", 11.0, True)], 0),
                self.row([("a", 11.0, True)], 1),
                self.row([("b", 11.0, True)], 2),
            ],
        )

        found = SheetHeadingFinder.scan(LayoutDocument(origin="cells", pages=[page], outline=[]))

        assert found.format == []

    def test__an_outline_entry_without_a_page_is_dropped(self) -> None:
        """A bookmark that lands nowhere corroborates nothing."""
        document = LayoutDocument(
            origin="cells", pages=[], outline=[OutlineEntry(title="NA", page=None, depth=1)]
        )

        assert SheetHeadingFinder.scan(document).outline == []
