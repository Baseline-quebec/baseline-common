"""Bindings — the plain bindings the hot loops call the library through, and their scratch."""

import ctypes
from typing import Any, ClassVar

import cffi
import pypdfium2_raw


class Bindings:
    """The library reached without the wrapper's typing, twice as quick, and its scratch cells."""

    # Each declaration is copied from the wrapper's generated bindings: a wrong one faults.
    FFI: ClassVar[Any] = cffi.FFI()
    FFI.cdef("""
        unsigned int FPDFText_GetUnicode(void *, int);
        int FPDFText_IsGenerated(void *, int);
        void *FPDFText_GetTextObject(void *, int);
        int FPDFText_GetCharOrigin(void *, int, double *, double *);
        int FPDFText_GetLooseCharBox(void *, int, float *);
        int FPDFPage_CountObjects(void *);
        void *FPDFPage_GetObject(void *, int);
        int FPDFFormObj_CountObjects(void *);
        void *FPDFFormObj_GetObject(void *, unsigned long);
        int FPDFPageObj_GetType(void *);
        int FPDFPageObj_GetBounds(void *, float *, float *, float *, float *);
        int FPDFPath_GetDrawMode(void *, int *, int *);
        int FPDFPageObj_GetStrokeWidth(void *, float *);
        int FPDFPageObj_GetFillColor(void *, unsigned int *, unsigned int *, unsigned int *,
                                     unsigned int *);
        int FPDFPageObj_GetStrokeColor(void *, unsigned int *, unsigned int *, unsigned int *,
                                       unsigned int *);
        unsigned long FPDFTextObj_GetText(void *, void *, unsigned short *, unsigned long);
        int FPDFTextObj_GetFontSize(void *, float *);
        typedef struct { float a, b, c, d, e, f; } FS_MATRIX;
        int FPDFPageObj_GetMatrix(void *, FS_MATRIX *);
    """)

    # SLF001: the wrapper's loaded handle is the one path that cannot open a second library.
    PDF: ClassVar[Any] = FFI.dlopen(pypdfium2_raw.bindings._libs["pdfium"]._name)  # noqa: SLF001

    # Shared scratch cells: safe only under ``LayoutParser.PDFIUM_LOCK``.
    BOX: ClassVar[Any] = FFI.new("float[4]")
    LEFT_P: ClassVar[Any] = FFI.new("float *")
    BOTTOM_P: ClassVar[Any] = FFI.new("float *")
    RIGHT_P: ClassVar[Any] = FFI.new("float *")
    TOP_P: ClassVar[Any] = FFI.new("float *")
    ORIGIN_X: ClassVar[Any] = FFI.new("double *")
    ORIGIN_Y: ClassVar[Any] = FFI.new("double *")
    FILL_P: ClassVar[Any] = FFI.new("int *")
    STROKED_P: ClassVar[Any] = FFI.new("int *")
    WIDTH_P: ClassVar[Any] = FFI.new("float *")
    RED: ClassVar[Any] = FFI.new("unsigned int *")
    GREEN: ClassVar[Any] = FFI.new("unsigned int *")
    BLUE: ClassVar[Any] = FFI.new("unsigned int *")
    ALPHA: ClassVar[Any] = FFI.new("unsigned int *")
    SIZE_P: ClassVar[Any] = FFI.new("float *")
    MATRIX: ClassVar[Any] = FFI.new("FS_MATRIX *")

    UTF16_NUL: ClassVar[int] = 2
    """Bytes the library spends on the terminator of a UTF-16 string it hands back."""

    @classmethod
    def handle(cls, raw: Any) -> object:  # noqa: ANN401
        """Hand one of the wrapper's typed handles to the plain bindings.

        Args:
            raw: A ctypes pointer the wrapper holds; the caller keeps its owner alive.

        Returns:
            The same address, untyped.
        """
        return cls.FFI.cast("void *", ctypes.cast(raw, ctypes.c_void_p).value)

    @classmethod
    def address(cls, pointer: object) -> int:
        """Read a pointer the plain bindings returned as a number, to compare it by identity.

        Args:
            pointer: A pointer the plain bindings returned.

        Returns:
            Its address.
        """
        return int(cls.FFI.cast("uintptr_t", pointer))
