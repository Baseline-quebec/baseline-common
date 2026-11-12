"""ObjectWalker — the object walk every reader of a page's paint shares."""

from collections.abc import Iterator
from typing import ClassVar

import pypdfium2.raw as pdfium_c

from baseline_core.core.llm.document_extraction.layout.source.bindings import Bindings


class ObjectWalker:
    """Walks a page's objects, descending into the forms it places."""

    FORM_DEPTH: ClassVar[int] = 15
    """How many forms deep the walk descends, as the library's own wrapper does."""

    # PLR0914: every library call is hoisted into a local, one lookup per million objects.
    @classmethod
    def walk(cls, page: object) -> Iterator[tuple[object, int]]:  # noqa: PLR0914
        """Walk a page's objects.

        Args:
            page: Raw handle of the page.

        Yields:
            (raw handle, type) of every object, the contents of placed forms included.
        """
        # An explicit stack: a drawing page holds a million objects, a frame per form is too many.
        pdf = Bindings.PDF
        count = pdf.FPDFPage_CountObjects(page)

        if count < 0:
            msg = "the library would not count the objects of the page"
            raise ValueError(msg)

        get_page_object = pdf.FPDFPage_GetObject
        get_form_object = pdf.FPDFFormObj_GetObject
        count_form = pdf.FPDFFormObj_CountObjects
        get_type = pdf.FPDFPageObj_GetType
        form = pdfium_c.FPDF_PAGEOBJ_FORM
        stack = [(page, count, 0, 0)]

        while stack:
            parent, total, index, level = stack.pop()
            get = get_page_object if level == 0 else get_form_object

            while index < total:
                obj = get(parent, index)
                index += 1

                if not obj:
                    msg = "the library would not hand back an object of the page"
                    raise ValueError(msg)

                kind = get_type(obj)
                yield obj, kind

                if kind == form and level < cls.FORM_DEPTH - 1:
                    stack.append((parent, total, index, level))
                    parent, total, index, level = obj, count_form(obj), 0, level + 1
                    get = get_form_object
