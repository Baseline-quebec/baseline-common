"""PaintReader — what a page paints beside text: rules, drawings and raster images."""

import math
from typing import Annotated

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    RasterImage,
    VectorDrawing,
)
from baseline_core.core.llm.document_extraction.layout.source.bindings import Bindings
from baseline_core.core.llm.document_extraction.layout.source.object_walker import ObjectWalker
from baseline_core.core.llm.document_extraction.layout.source.painted_path import PaintedPath


class PaintConfig(BaseModel):
    """Where a painted path stops being a rule and starts being a drawing."""

    model_config = ConfigDict(frozen=True)

    vector_min_side: Annotated[
        float,
        Field(
            description="Vector paths thinner than this (1/4 inch) are rules and underlines, "
            "not drawings."
        ),
    ] = 18.0
    vector_cluster_cap: Annotated[
        int,
        Field(
            description="Paths above which a page's drawings surface as ONE union image instead "
            "of one per path; no drawing is dropped, just unsplit. A full-page plan carries "
            "tens of thousands."
        ),
    ] = 2_000
    rule_ceiling: Annotated[
        float,
        Field(
            description="Height, in points, above which a mark rules nothing: it would underline "
            "text ninety-six points tall."
        ),
    ] = 24.0
    white: Annotated[
        int,
        Field(
            description="Colour component over which, on all three channels, paint is white on "
            "white paper: a table ruled in white underlines nothing."
        ),
    ] = 245


class PaintReader:
    """Reads what a page paints beside text, in one pass over its objects."""

    def __init__(self, config: PaintConfig | None = None) -> None:
        """Bind the paint thresholds.

        Args:
            config: Where a path stops being a rule and starts being a drawing.
        """
        self._config = config or PaintConfig()

    # PLR0914: the scratch cells and thresholds are hoisted into locals, one lookup per path.
    def read(  # noqa: PLR0914
        self, page: pdfium.PdfPage, height: float
    ) -> tuple[list[PaintedPath], list[VectorDrawing], list[RasterImage]]:
        """Read what the page paints beside text.

        Args:
            page: The library's page object.
            height: Page height, to flip the library's bottom-up coordinates.

        Returns:
            (rules, drawings, rasters), the three things the rest of the parse reads.
        """
        pdf = Bindings.PDF
        left, bottom, right, top = (
            Bindings.LEFT_P,
            Bindings.BOTTOM_P,
            Bindings.RIGHT_P,
            Bindings.TOP_P,
        )
        min_side, ceiling = self._config.vector_min_side, self._config.rule_ceiling
        rules: list[PaintedPath] = []
        drawings: list[BBox] = []
        rasters: list[RasterImage] = []
        paths = 0
        union = [math.inf, math.inf, -math.inf, -math.inf]

        for obj, kind in ObjectWalker.walk(Bindings.handle(page.raw)):
            if kind == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                rasters.append(self._raster(obj, height))
                continue

            if kind != pdfium_c.FPDF_PAGEOBJ_PATH:
                continue

            pdf.FPDFPageObj_GetBounds(obj, left, bottom, right, top)
            x0, x1 = left[0], right[0]

            if x1 < x0:
                x0, x1 = x1, x0

            y0, y1 = height - top[0], height - bottom[0]

            if y1 < y0:
                y0, y1 = y1, y0

            paths += 1
            union[0], union[1] = min(union[0], x0), min(union[1], y0)
            union[2], union[3] = max(union[2], x1), max(union[3], y1)
            wide, tall = x1 - x0, y1 - y0

            if wide >= min_side and tall >= min_side:
                drawings.append(BBox(x0=x0, y0=y0, x1=x1, y1=y1))

            if wide > tall and tall <= ceiling:
                rule = self._rule(obj, x0, y0, x1, y1)

                if rule is not None:
                    rules.append(rule)

        if paths > self._config.vector_cluster_cap:
            drawings = [BBox(x0=union[0], y0=union[1], x1=union[2], y1=union[3])]

        return rules, [VectorDrawing(bbox=box) for box in drawings], rasters

    @staticmethod
    def _raster(obj: object, height: float) -> RasterImage:
        """Read one placed image: the extent it covers.

        Args:
            obj: Raw handle of the page object, of image type.
            height: Page height, to flip the library's bottom-up coordinates.

        Returns:
            The placement.
        """
        left, bottom, right, top = (
            Bindings.LEFT_P,
            Bindings.BOTTOM_P,
            Bindings.RIGHT_P,
            Bindings.TOP_P,
        )
        Bindings.PDF.FPDFPageObj_GetBounds(obj, left, bottom, right, top)
        return RasterImage(
            bbox=BBox(
                x0=min(left[0], right[0]),
                y0=height - max(bottom[0], top[0]),
                x1=max(left[0], right[0]),
                y1=height - min(bottom[0], top[0]),
            )
        )

    def _rule(self, obj: object, x0: float, y0: float, x1: float, y1: float) -> PaintedPath | None:
        """Read a path as a rule, if it can rule a line at all.

        Args:
            obj: Raw handle of the page object, of path type.
            x0: Left edge, already flipped to top-down points.
            y0: Top edge.
            x1: Right edge.
            y1: Bottom edge.

        Returns:
            The mark, or None when the paint that draws it cannot be seen.
        """
        pdf = Bindings.PDF
        pdf.FPDFPath_GetDrawMode(obj, Bindings.FILL_P, Bindings.STROKED_P)
        stroked = Bindings.STROKED_P[0]

        if not self._visible(obj, filled=Bindings.FILL_P[0] != 0, stroked=bool(stroked)):
            return None

        if not stroked:
            return PaintedPath(x0=x0, y0=y0, x1=x1, y1=y1, thickness=y1 - y0)

        # The library inflates a stroke's bounds by its width on every side; taken back off.
        pdf.FPDFPageObj_GetStrokeWidth(obj, Bindings.WIDTH_P)
        width = abs(Bindings.WIDTH_P[0])
        return PaintedPath(
            x0=x0, y0=y0, x1=x1, y1=y1, thickness=max(0.0, (y1 - y0) - 2 * width) + width
        )

    def _visible(self, obj: object, *, filled: bool, stroked: bool) -> bool:
        """Tell whether a path's paint marks the page at all.

        Args:
            obj: Raw handle of the page object, of path type.
            filled: Whether the path is filled.
            stroked: Whether the path is stroked.

        Returns:
            True when some paint that draws it is neither transparent nor white.
        """
        pdf, white = Bindings.PDF, self._config.white
        red, green, blue, alpha = Bindings.RED, Bindings.GREEN, Bindings.BLUE, Bindings.ALPHA

        if filled:
            pdf.FPDFPageObj_GetFillColor(obj, red, green, blue, alpha)

            if alpha[0] and not (red[0] > white and green[0] > white and blue[0] > white):
                return True

        if stroked:
            pdf.FPDFPageObj_GetStrokeColor(obj, red, green, blue, alpha)

            if alpha[0] and not (red[0] > white and green[0] > white and blue[0] > white):
                return True

        return False
