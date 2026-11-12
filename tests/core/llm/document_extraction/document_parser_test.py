from baseline_core.core.llm.document_extraction.document_parser import DocumentParser


class TestDocumentParser:
    def test__a_pdf__is_read_as_pages(self, no_outline_bytes: bytes) -> None:
        """A PDF goes to the layout parser and comes back as a page tree."""
        document = DocumentParser().parse(no_outline_bytes)

        assert document.origin == "points"
        assert document.pages

    def test__a_workbook__is_read_as_cells(self, workbook_bytes: bytes) -> None:
        """A workbook is recognised by its bytes and read as a cell tree."""
        document = DocumentParser().parse(workbook_bytes)

        assert document.origin == "cells"
        assert document.pages
