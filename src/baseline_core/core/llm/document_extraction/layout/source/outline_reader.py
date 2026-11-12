"""OutlineReader — the embedded outline of an open document."""

import pypdfium2 as pdfium

from baseline_core.core.llm.document_extraction.layout.layout import OutlineEntry


class OutlineReader:
    """Reads the embedded outline of an open document."""

    @staticmethod
    def read(doc: pdfium.PdfDocument) -> list[OutlineEntry]:
        """Read the embedded outline.

        Args:
            doc: The open document.

        Returns:
            One entry per bookmark, in outline order.
        """
        entries: list[OutlineEntry] = []

        for bookmark in doc.get_toc():
            destination = bookmark.get_dest()
            page_index = destination.get_index() if destination is not None else None
            entries.append(
                OutlineEntry(
                    title=bookmark.get_title(),
                    page=page_index + 1 if page_index is not None else None,
                    depth=bookmark.level + 1,
                )
            )

        return entries
