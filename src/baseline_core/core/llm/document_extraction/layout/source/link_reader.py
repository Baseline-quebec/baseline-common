"""LinkReader — a page's link annotations, the outbound ones kept."""

import ctypes

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from baseline_core.core.llm.document_extraction.layout.layout import BBox, Link


class LinkReader:
    """Reads a page's link annotations, keeping the ones that point outside the document."""

    @classmethod
    def read(cls, doc: pdfium.PdfDocument, page: pdfium.PdfPage, height: float) -> list[Link]:
        """Read the page's link annotations, keeping the ones that point outside.

        Args:
            doc: The open document, which URI actions resolve against.
            page: The library's page object.
            height: Page height, to flip the library's bottom-up coordinates.

        Returns:
            One entry per outbound link (URI, GoToR, Launch), in the document's own order.
        """
        links: list[Link] = []
        position = ctypes.c_int(0)
        link = pdfium_c.FPDF_LINK()
        rect = pdfium_c.FS_RECTF()

        while pdfium_c.FPDFLink_Enumerate(page, ctypes.byref(position), ctypes.byref(link)):
            action = pdfium_c.FPDFLink_GetAction(link)

            if not action:
                continue

            kind = pdfium_c.FPDFAction_GetType(action)

            if kind == pdfium_c.PDFACTION_URI:
                target = cls._uri(doc, action)
            elif kind in {pdfium_c.PDFACTION_REMOTEGOTO, pdfium_c.PDFACTION_LAUNCH}:
                target = cls._file_path(action)
            else:
                target = None

            if not target:
                continue

            pdfium_c.FPDFLink_GetAnnotRect(link, ctypes.byref(rect))
            links.append(
                Link(
                    target=target,
                    bbox=BBox(
                        x0=min(rect.left, rect.right),
                        y0=height - max(rect.top, rect.bottom),
                        x1=max(rect.left, rect.right),
                        y1=height - min(rect.top, rect.bottom),
                    ),
                )
            )

        return links

    @staticmethod
    def _uri(doc: pdfium.PdfDocument, action: object) -> str | None:
        """Read a URI action's destination.

        Args:
            doc: The open document.
            action: The library's action handle.

        Returns:
            The URI, or None when the action names none.
        """
        length = pdfium_c.FPDFAction_GetURIPath(doc, action, None, 0)

        if length <= 1:
            return None

        buffer = ctypes.create_string_buffer(length)
        pdfium_c.FPDFAction_GetURIPath(doc, action, buffer, length)
        return buffer.raw[: length - 1].decode("utf-8", "ignore") or None

    @staticmethod
    def _file_path(action: object) -> str | None:
        """Read a GoToR or Launch action's file path.

        Args:
            action: The library's action handle.

        Returns:
            The path, or None when the action names none.
        """
        length = pdfium_c.FPDFAction_GetFilePath(action, None, 0)

        if length <= 1:
            return None

        buffer = ctypes.create_string_buffer(length)
        pdfium_c.FPDFAction_GetFilePath(action, buffer, length)
        return buffer.raw[: length - 1].decode("utf-8", "ignore") or None
