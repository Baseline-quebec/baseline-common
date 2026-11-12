"""DocumentParser: the one place a file's format is read, a workbook or a PDF."""

from baseline_core.core.llm.document_extraction.layout.layout import LayoutDocument
from baseline_core.core.llm.document_extraction.layout.layout_parser import (
    LayoutConfig,
    LayoutParser,
)
from baseline_core.core.llm.document_extraction.sheet_parser import SheetParser


class DocumentParser:
    """Turns the bytes of a file into the layout tree every stage reads, whatever its format."""

    def __init__(self, config: LayoutConfig | None = None) -> None:
        """Build the PDF parser with the project's settings; a workbook needs none.

        Args:
            config: The PDF parse settings (header, footer, columns); the defaults when None.
        """
        self._layout = LayoutParser(config)

    def parse(self, content: bytes) -> LayoutDocument:
        """Parse a file into its layout tree.

        Args:
            content: The file's bytes.

        Returns:
            A workbook's tree (``origin="cells"``) when the bytes are one, else a page tree.
        """
        if SheetParser.recognizes(content):
            return SheetParser.parse(content)

        return self._layout.parse(content)
