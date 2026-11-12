"""SheetParser — a workbook read into the layout tree, and the only place its library is named."""

import hashlib
import itertools
import zipfile
from io import BytesIO
from typing import ClassVar

import openpyxl
from loguru import logger
from openpyxl.cell.cell import Cell
from openpyxl.utils import column_index_from_string
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

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


class SheetParser:
    """Read a workbook into the tree the pipeline speaks, one visible sheet to a page."""

    DEFAULT_SIZE: ClassVar[float] = 11.0
    """Points a face is set at when the cell does not say."""

    DEFAULT_ROW_HEIGHT: ClassVar[float] = 15.0
    """Points a row stands when the sheet declares no default of its own."""

    DEFAULT_COLUMN_WIDTH: ClassVar[float] = 8.43
    """Characters a column runs when the sheet declares nothing for it."""

    PIXELS_PER_CHAR: ClassVar[int] = 7
    """Pixels per character of column width, plus ``COLUMN_PADDING`` — the spreadsheet rule."""

    COLUMN_PADDING: ClassVar[int] = 5
    """Pixels a column adds beside its characters."""

    POINTS_PER_PIXEL: ClassVar[float] = 0.75
    """Points per screen pixel at the 96 dpi a sheet is laid out at."""

    WORKBOOK_PART: ClassVar[str] = "xl/workbook.xml"
    """The archive member only an Office workbook carries."""

    @classmethod
    def recognizes(cls, content: bytes) -> bool:
        """Tell whether the bytes are a workbook this parser reads.

        Args:
            content: Raw bytes of a file.

        Returns:
            True for an Office workbook archive, False for anything else.
        """
        buffer = BytesIO(content)

        if not zipfile.is_zipfile(buffer):
            return False

        try:
            with zipfile.ZipFile(buffer) as archive:
                return cls.WORKBOOK_PART in archive.namelist()
        except zipfile.BadZipFile:
            return False

    @classmethod
    def parse(cls, content: bytes) -> LayoutDocument:
        """Read one workbook.

        Args:
            content: Raw bytes of the workbook.

        Returns:
            The document, hidden and empty sheets dropped, each sheet's name an outline entry.
        """
        book = openpyxl.load_workbook(BytesIO(content), data_only=True)
        uncached = cls._uncached_formulas(content, book)

        if uncached:
            # A workbook written by a tool that never calculated has formulas and no values;
            # without this count it would read as blanks and no error.
            logger.warning("{} formula cells carry no cached value and read as blank", uncached)

        visible = [title for title in book.sheetnames if book[title].sheet_state == "visible"]
        read = [
            (cls._page(book[title], number), title)
            for number, title in enumerate(visible, start=1)
        ]
        kept = [(page, title) for page, title in read if page is not None]
        return LayoutDocument(
            origin="cells",
            pages=[page for page, _ in kept],
            outline=[OutlineEntry(title=title, page=page.number, depth=1) for page, title in kept],
            checksum=hashlib.sha256(content).hexdigest(),
        )

    @classmethod
    def _uncached_formulas(cls, content: bytes, cached: Workbook) -> int:
        """Count the formula cells the application never cached a value for.

        Args:
            content: Raw bytes of the workbook.
            cached: The same workbook, loaded for its cached values.

        Returns:
            How many formulas read as blank.
        """
        # The library hands back formulas OR values, never both, so the file is read twice.
        formulas = openpyxl.load_workbook(BytesIO(content), read_only=True)

        try:
            return sum(
                cell.data_type == "f" and cached[title][cell.coordinate].value is None
                for title in formulas.sheetnames
                for row in formulas[title].iter_rows()
                for cell in row
            )
        finally:
            formulas.close()

    @classmethod
    def _page(cls, sheet: Worksheet, number: int) -> LayoutPage | None:
        """Read one sheet as one page, blank rows splitting its blocks.

        Args:
            sheet: The sheet.
            number: Its place among the visible sheets, one-based.

        Returns:
            The page, in points from the sheet's row heights and column widths; None when blank.
        """
        tops = cls._row_edges(sheet)
        lefts = cls._column_edges(sheet)
        blocks: list[list[LayoutLine]] = []

        for row in sheet.iter_rows():
            line = cls._line(row, tops, lefts)

            if line is None:
                blocks.append([])
                continue

            if not blocks:
                blocks.append([])

            blocks[-1].append(line)

        built = [TextBlock(lines=lines) for lines in blocks if lines]

        if not built:
            return None

        return LayoutPage(
            number=number, size=PageSize(width=lefts[-1], height=tops[-1]), blocks=built
        )

    @classmethod
    def _row_edges(cls, sheet: Worksheet) -> list[float]:
        """Lay the rows out top-down.

        Args:
            sheet: The sheet.

        Returns:
            The top edge of every row in points, then the bottom edge of the last.
        """
        default = sheet.sheet_format.defaultRowHeight or cls.DEFAULT_ROW_HEIGHT
        heights = [
            (sheet.row_dimensions[index].height or default)
            if index in sheet.row_dimensions
            else default
            for index in range(1, sheet.max_row + 1)
        ]
        return [0.0, *itertools.accumulate(heights)]

    @classmethod
    def _column_edges(cls, sheet: Worksheet) -> list[float]:
        """Lay the columns out left to right.

        Args:
            sheet: The sheet.

        Returns:
            The left edge of every column in points, then the right edge of the last.
        """
        default = sheet.sheet_format.defaultColWidth or cls.DEFAULT_COLUMN_WIDTH
        chars = [default] * sheet.max_column

        # A declared width may cover a run of columns, keyed by the first of them.
        for key, dimension in sheet.column_dimensions.items():
            if not dimension.customWidth or dimension.width is None:
                continue

            first = dimension.min or column_index_from_string(key)

            for index in range(first, min(dimension.max or first, sheet.max_column) + 1):
                chars[index - 1] = dimension.width

        widths = [
            (width * cls.PIXELS_PER_CHAR + cls.COLUMN_PADDING) * cls.POINTS_PER_PIXEL
            for width in chars
        ]
        return [0.0, *itertools.accumulate(widths)]

    @classmethod
    def _line(
        cls, row: tuple[Cell, ...], tops: list[float], lefts: list[float]
    ) -> LayoutLine | None:
        """Read one row as one line, its non-empty cells joined in column order.

        Args:
            row: The row's cells.
            tops: The row edges, from ``_row_edges``.
            lefts: The column edges, from ``_column_edges``.

        Returns:
            The line, or None when the row prints nothing.
        """
        spans: list[LayoutSpan] = []
        columns: list[int] = []

        for cell in row:
            text = "" if cell.value is None else " ".join(str(cell.value).split())

            if not text:
                continue

            font = cell.font
            spans.append(
                LayoutSpan(
                    text=text,
                    source=cell.coordinate,
                    size=float(font.size if font and font.size else cls.DEFAULT_SIZE),
                    font=(font.name if font and font.name else "") or "",
                    flags=(LayoutSpan.BOLD if font and font.bold else 0)
                    | (LayoutSpan.ITALIC if font and font.italic else 0),
                )
            )
            columns.append(cell.column)

        if not spans:
            return None

        index = row[0].row
        return LayoutLine(
            text=" ".join(span.text for span in spans),
            bbox=BBox(
                x0=lefts[columns[0] - 1], y0=tops[index - 1], x1=lefts[columns[-1]], y1=tops[index]
            ),
            spans=spans,
        )
