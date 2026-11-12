"""LayoutParser — the one door to the library for parsing: bytes in, the tree out."""

import hashlib
import threading
from typing import ClassVar

import pypdfium2 as pdfium
from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.assembly.block_builder import BlockBuilder
from baseline_core.core.llm.document_extraction.layout.assembly.furniture_marker import (
    FurnitureMarker,
    HeaderFooterConfig,
)
from baseline_core.core.llm.document_extraction.layout.assembly.underline_marker import (
    UnderlineMarker,
)
from baseline_core.core.llm.document_extraction.layout.column_reorder import (
    ColumnConfig,
    ColumnReorder,
)
from baseline_core.core.llm.document_extraction.layout.layout import (
    LayoutDocument,
    LayoutPage,
    PageSize,
)
from baseline_core.core.llm.document_extraction.layout.source.advances import Advances
from baseline_core.core.llm.document_extraction.layout.source.char_reader import CharReader
from baseline_core.core.llm.document_extraction.layout.source.link_reader import LinkReader
from baseline_core.core.llm.document_extraction.layout.source.outline_reader import OutlineReader
from baseline_core.core.llm.document_extraction.layout.source.paint_reader import PaintReader


class LayoutConfig(BaseModel):
    """The settings a project changes in a PDF parse: the running header and footer, the columns."""

    model_config = ConfigDict(frozen=True)

    header_footer: HeaderFooterConfig = Field(default_factory=HeaderFooterConfig)
    columns: ColumnConfig = Field(default_factory=ColumnConfig)


class LayoutParser:
    """Parses a PDF into pages, lines, runs and bookmarks."""

    # PDFium is not thread-safe and holds no internal mutex of its own: concurrent calls segfault.
    PDFIUM_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: LayoutConfig | None = None) -> None:
        """Build the readers and builders a parse runs through, with the project's settings.

        Args:
            config: The header, footer and column settings; the defaults when None.
        """
        config = config or LayoutConfig()
        self._chars = CharReader()
        self._advances = Advances()
        self._paint = PaintReader()
        self._links = LinkReader()
        self._outline = OutlineReader()
        self._underline = UnderlineMarker()
        self._structure = BlockBuilder()
        self._furniture = FurnitureMarker(config.header_footer)
        self._columns = ColumnReorder(config.columns)

    def parse(self, pdf: bytes) -> LayoutDocument:
        """Parse a document into pages, lines, runs and bookmarks.

        Args:
            pdf: Raw bytes of the document.

        Returns:
            The parsed tree in reading order, each page told which of its edge lines is furniture.
        """
        with self.PDFIUM_LOCK:
            doc = pdfium.PdfDocument(pdf)

            try:
                pages = [self._page(doc, index) for index in range(len(doc))]
                outline = self._outline.read(doc)
            finally:
                doc.close()

        # Furniture first: its verdict is voted on each page's top and bottom line by position,
        # which no reorder moves, and the reorder then carries it over on every page it copies.
        return self._columns.reorder(
            LayoutDocument(
                pages=self._furniture.mark(pages),
                outline=outline,
                checksum=hashlib.sha256(pdf).hexdigest(),
            )
        )

    def _page(self, doc: pdfium.PdfDocument, index: int) -> LayoutPage:
        """Parse one page: its text blocks, its painted images, and its link annotations.

        Args:
            doc: The open document.
            index: 0-based physical page index.

        Returns:
            The page.
        """
        page = doc[index]
        width, height = page.get_size()
        # A height of zero would make every position on the page infinite.
        height = height or 1.0
        textpage = page.get_textpage()
        chars = self._advances.apply(self._chars.read(page, textpage, height))
        rules, drawings, rasters = self._paint.read(page, height)
        self._underline.mark(chars, rules)
        return LayoutPage(
            number=index + 1,
            size=PageSize(width=float(width), height=float(height)),
            blocks=self._structure.build(chars),
            links=self._links.read(doc, page, height),
            images=[*rasters, *drawings],
        )
