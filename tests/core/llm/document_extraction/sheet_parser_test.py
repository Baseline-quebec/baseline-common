"""Unit tests for the workbook parser: sheets as pages, rows as lines, laid out in points."""

import zipfile
from io import BytesIO

import openpyxl
import pytest
from loguru import logger

from baseline_core.core.llm.document_extraction.sheet_parser import SheetParser


class TestRecognizes:
    def test__a_workbook_is_recognized(self, workbook_bytes: bytes) -> None:
        """An Office workbook archive is claimed by the sheet parser."""
        assert SheetParser.recognizes(workbook_bytes) is True

    def test__a_pdf_is_not(self, no_outline_bytes: bytes) -> None:
        """PDF bytes fall through to the page parser."""
        assert SheetParser.recognizes(no_outline_bytes) is False

    def test__another_office_archive_is_not(self) -> None:
        """A zip without a workbook part, such as a Word document, is not a workbook."""
        buffer = BytesIO()

        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", "<w:document/>")

        assert SheetParser.recognizes(buffer.getvalue()) is False


class TestParse:
    def test__one_page_per_visible_sheet_that_prints(self, workbook_bytes: bytes) -> None:
        """The hidden sheet and the empty one are dropped; the others keep their tab position."""
        document = SheetParser.parse(workbook_bytes)
        assert [page.number for page in document.pages] == [1, 3]
        assert document.origin == "cells"
        assert document.checksum is not None

    def test__sheet_names_travel_as_outline_entries(self, workbook_bytes: bytes) -> None:
        """Each kept sheet's name becomes a top-level bookmark pointing at its page."""
        document = SheetParser.parse(workbook_bytes)
        assert [(entry.title, entry.page, entry.depth) for entry in document.outline] == [
            ("Bilan", 1, 1),
            ("Notes", 3, 1),
        ]

    def test__a_row_is_one_line_of_its_cells_in_column_order(self, workbook_bytes: bytes) -> None:
        """Non-empty cells join into one line, whitespace and newlines collapsed."""
        page = SheetParser.parse(workbook_bytes).pages[0]
        line = page.lines[3]
        assert line.text == "Encaisse et placements 100 200"
        assert [span.source for span in line.spans] == ["A5", "B5", "C5"]

    def test__blank_rows_split_blocks(self, workbook_bytes: bytes) -> None:
        """Row 2 and row 7 are blank, so the sheet reads as three blocks."""
        page = SheetParser.parse(workbook_bytes).pages[0]
        assert [block.text for block in page.blocks] == [
            "Situation financière",
            "Poste 20XX 20XY Actifs financiers Encaisse et placements 100 200 Placements 50 60",
            "Total 300",
        ]

    def test__a_cell_keeps_its_face(self, workbook_bytes: bytes) -> None:
        """Size, bold and italic come off the cell's font; the default face is 11 points."""
        page = SheetParser.parse(workbook_bytes).pages[0]
        title, _, _, body = page.lines[:4]
        assert (title.spans[0].size, title.spans[0].bold) == (14.0, True)
        assert (body.spans[0].size, body.spans[0].italic) == (11.0, True)

    def test__rows_and_columns_are_laid_out_in_points(self, workbook_bytes: bytes) -> None:
        """Default rows stand 15 points; a column's declared width sets its edge in points."""
        page = SheetParser.parse(workbook_bytes).pages[0]
        line = page.lines[3]
        default_column = (8.43 * 7 + 5) * 0.75
        wide_column = (20 * 7 + 5) * 0.75
        assert (line.bbox.y0, line.bbox.y1) == (60.0, 75.0)
        assert line.bbox.x0 == 0.0
        assert line.bbox.x1 == pytest.approx(2 * default_column + wide_column)
        assert page.size.width == pytest.approx(2 * default_column + wide_column)
        assert page.size.height == 120.0

    def test__a_formula_never_calculated_reads_blank_and_is_counted(self) -> None:
        """The library computes nothing: an uncached formula is dropped, and the drop is logged."""
        book = openpyxl.Workbook()
        sheet = book.active
        sheet["A1"], sheet["A2"], sheet["A3"] = 5, "=A1*2", "Total"
        buffer = BytesIO()
        book.save(buffer)
        warnings: list[str] = []
        handle = logger.add(warnings.append, level="WARNING", format="{message}")

        try:
            page = SheetParser.parse(buffer.getvalue()).pages[0]
        finally:
            logger.remove(handle)

        assert [line.text for line in page.lines] == ["5", "Total"]
        assert [w.strip() for w in warnings] == [
            "1 formula cells carry no cached value and read as blank"
        ]
