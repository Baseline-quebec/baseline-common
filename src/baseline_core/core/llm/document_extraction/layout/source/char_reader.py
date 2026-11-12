"""CharReader — every character of a text page, with its geometry and its face."""

import ctypes
import math
from typing import Annotated

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.source.bindings import Bindings
from baseline_core.core.llm.document_extraction.layout.source.char import Char
from baseline_core.core.llm.document_extraction.layout.source.object_walker import ObjectWalker
from baseline_core.core.llm.document_extraction.layout.source.style import Style


class CharConfig(BaseModel):
    """What the character reader assumes where the file declares nothing."""

    model_config = ConfigDict(frozen=True)

    unmapped_char: Annotated[
        str,
        Field(
            description="What a character the file fails to map to Unicode reads as: a subset "
            "face that loses an entry usually loses its hyphen, « Nord\x02Est »."
        ),
    ] = "-"
    default_ascender: Annotated[
        float,
        Field(description="Ascent per point of size to assume of a face that declares none."),
    ] = 0.9


class CharReader:
    """Reads every character of a text page, styles cached per text object."""

    def __init__(self, config: CharConfig | None = None) -> None:
        """Bind the reader's defaults.

        Args:
            config: What to assume where the file declares nothing.
        """
        self._config = config or CharConfig()

    def read(
        self, page: pdfium.PdfPage, textpage: pdfium.PdfTextPage, height: float
    ) -> list[Char]:
        """Read every character with its geometry and style, in the order the file paints them.

        Args:
            page: The library's page object.
            textpage: Its text page.
            height: Page height, to flip the library's bottom-up coordinates.

        Returns:
            The chars in painting order, the spaces the library invents left out, widths owed.
        """
        # The text page drops an object that paints only a space, and that space is the blank
        # line a paragraph is set off by: put back where it is painted, it moves the pen.
        width = float(page.get_size()[0])
        painted = self._painted(textpage, width, height)
        text_page = Bindings.handle(textpage.raw)
        chars: list[Char] = []

        for obj, kind in ObjectWalker.walk(Bindings.handle(page.raw)):
            if kind != pdfium_c.FPDF_PAGEOBJ_TEXT:
                continue

            group = painted.pop(Bindings.address(obj), None)

            if group is not None:
                chars.extend(group)
            elif not self._object_text(obj, text_page).strip():
                blank = self._blank(obj, height)

                # A space set off the page prints nothing, so it must not move the pen.
                if 0 <= blank.baseline <= height and 0 <= blank.x0 <= width:
                    chars.append(blank)

        for group in painted.values():
            chars.extend(group)

        return chars

    def _painted(
        self, textpage: pdfium.PdfTextPage, width: float, height: float
    ) -> dict[int, list[Char]]:
        """Read the characters the text page kept, grouped by the text object painting them.

        Args:
            textpage: The library's text page.
            width: Page width, the right edge of what is printed.
            height: Page height, to flip the library's bottom-up coordinates.

        Returns:
            The chars of each text object in the order the page reads them, keyed by object.
        """
        pdf, box = Bindings.PDF, Bindings.BOX
        origin_x, origin_y = Bindings.ORIGIN_X, Bindings.ORIGIN_Y
        unmapped = self._config.unmapped_char
        chars: dict[int, list[Char]] = {}
        styles: dict[int | None, Style] = {}
        after_jump = False
        painter = -1
        style = Style(font="", size=0.0, weight=0, descriptor=0, ascender=0.0)
        pages = Bindings.handle(textpage.raw)

        for index in range(textpage.count_chars()):
            code = pdf.FPDFText_GetUnicode(pages, index)
            ch = chr(code) if code else ""

            if not ch or ch in "\r\n":
                continue

            # A control code is a subset face whose ToUnicode lost its hyphen.
            if ch < " ":
                ch = unmapped

            if not ch.strip() and pdf.FPDFText_IsGenerated(pages, index):
                after_jump = True
                continue

            address = Bindings.address(pdf.FPDFText_GetTextObject(pages, index))

            if address != painter:
                painter = address

                if address not in styles:
                    styles[address] = self._style(
                        textpage, index, pdfium_c.FPDFText_GetTextObject(textpage, index)
                    )

                style = styles[address]

            pdf.FPDFText_GetCharOrigin(pages, index, origin_x, origin_y)

            if not (0 <= origin_x[0] <= width and 0 <= origin_y[0] <= height):
                after_jump = False
                continue

            pdf.FPDFText_GetLooseCharBox(pages, index, box)
            chars.setdefault(address, []).append(
                Char(
                    ch=ch,
                    x0=origin_x[0],
                    y0=height - max(box[1], box[3]),
                    x1=max(box[0], box[2]),
                    y1=height - min(box[1], box[3]),
                    baseline=height - origin_y[0],
                    style=style,
                    after_jump=after_jump,
                )
            )
            after_jump = False

        return chars

    def _style(self, textpage: pdfium.PdfTextPage, index: int, text_object: object) -> Style:
        """Read the style the text object painting one character declares.

        Args:
            textpage: The library's text page.
            index: 0-based index of the character on that page.
            text_object: The library's handle for the text object painting it; falsy when none.

        Returns:
            The style, its ascent defaulted when the face declares none.
        """
        name = ctypes.create_string_buffer(256)
        descriptor = ctypes.c_int()
        length = pdfium_c.FPDFText_GetFontInfo(
            textpage, index, name, 256, ctypes.byref(descriptor)
        )
        matrix = pdfium_c.FS_MATRIX()
        pdfium_c.FPDFText_GetMatrix(textpage, index, ctypes.byref(matrix))
        return Style(
            font=name.raw[: max(0, length - 1)].decode("utf-8", "ignore"),
            # Text-space size through the WHOLE matrix: a condensed face is the area it covers.
            size=round(
                pdfium_c.FPDFText_GetFontSize(textpage, index) * (self._scale(matrix) or 1), 2
            ),
            **self._direction(matrix),
            weight=pdfium_c.FPDFText_GetFontWeight(textpage, index),
            descriptor=descriptor.value,
            ascender=self._ascender(text_object),
        )

    @staticmethod
    def _direction(matrix: pdfium_c.FS_MATRIX) -> dict[str, float]:
        """Unit vector the run is written along, flipped to top-down points.

        Args:
            matrix: The library's text matrix for one character.

        Returns:
            Its ``dx`` and ``dy``, rounded so one run answers with one direction.
        """
        reach = math.hypot(matrix.a, matrix.b)

        if not reach:
            return {"dx": 1.0, "dy": 0.0}

        return {"dx": round(matrix.a / reach, 6), "dy": round(-matrix.b / reach, 6)}

    @staticmethod
    def _scale(matrix: pdfium_c.FS_MATRIX) -> float:
        """Scale a text matrix applies, as one number.

        Args:
            matrix: The library's text matrix for one character.

        Returns:
            The square root of the area the matrix scales a unit square to.
        """
        return math.sqrt(abs(matrix.a * matrix.d - matrix.b * matrix.c))

    def _ascender(self, text_object: object) -> float:
        """Read the face's ascent, as a share of its size.

        Args:
            text_object: The library's text object handle; falsy when the character reports none.

        Returns:
            The ascent per point of size, or the default when the face declares none.
        """
        default = self._config.default_ascender

        if not text_object:
            return default

        font = pdfium_c.FPDFTextObj_GetFont(text_object)

        if not font:
            return default

        ascent = ctypes.c_float()

        if not pdfium_c.FPDFFont_GetAscent(font, 1.0, ctypes.byref(ascent)):
            return default

        return abs(ascent.value) or default

    def _blank(self, obj: object, height: float) -> Char:
        """Read a text object that paints only a space, as the space it paints.

        Args:
            obj: Raw handle of the text object.
            height: Page height, to flip the library's bottom-up coordinates.

        Returns:
            One space as wide as the object paints, on its baseline since it carries no ink.
        """
        pdf = Bindings.PDF
        left, bottom, right, top = (
            Bindings.LEFT_P,
            Bindings.BOTTOM_P,
            Bindings.RIGHT_P,
            Bindings.TOP_P,
        )
        pdf.FPDFPageObj_GetBounds(obj, left, bottom, right, top)
        pdf.FPDFTextObj_GetFontSize(obj, Bindings.SIZE_P)
        pdf.FPDFPageObj_GetMatrix(obj, Bindings.MATRIX)
        matrix = Bindings.MATRIX[0]
        x0, x1 = min(left[0], right[0]), max(left[0], right[0])
        y0, y1 = height - max(bottom[0], top[0]), height - min(bottom[0], top[0])
        direction = self._direction(matrix)
        style = Style(
            font="",
            size=round(Bindings.SIZE_P[0] * (self._scale(matrix) or 1), 2),
            **direction,
            weight=0,
            descriptor=0,
            ascender=self._config.default_ascender,
        )
        return Char(
            ch=" ",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            baseline=y1,
            style=style,
            advance=(x1 - x0) * abs(direction["dx"]) + (y1 - y0) * abs(direction["dy"]),
            blank=True,
        )

    @staticmethod
    def _object_text(obj: object, textpage: object) -> str:
        """Read what one text object paints, as the text page reads it.

        Args:
            obj: Raw handle of a text object.
            textpage: Raw handle of the text page of the same page.

        Returns:
            Its characters, empty when the library kept none of them.
        """
        ffi, pdf, nul = Bindings.FFI, Bindings.PDF, Bindings.UTF16_NUL
        size = pdf.FPDFTextObj_GetText(obj, textpage, ffi.NULL, 0)

        if size <= nul:
            return ""

        buffer = ffi.new("unsigned short[]", size // 2 + 1)
        pdf.FPDFTextObj_GetText(obj, textpage, buffer, size)
        raw: bytes = ffi.buffer(buffer, size - nul)[:]
        return raw.decode("utf-16-le", "ignore")
