"""The frozen layout tree: pages, blocks, lines, runs and bookmarks, with no library behind it."""

from collections import Counter
from collections.abc import Iterable
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from baseline_core.core.llm.document_extraction.text_search import TextSearch


class BBox(BaseModel):
    """Position of a text run on its page, in PDF points from the top-left corner."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def union(cls, boxes: Iterable["BBox"]) -> "BBox | None":
        """Return the extent covering every box given.

        Args:
            boxes: The extents to cover.

        Returns:
            Their bounding extent, or None when none was given.
        """
        covered = list(boxes)

        if not covered:
            return None

        return cls(
            x0=min(box.x0 for box in covered),
            y0=min(box.y0 for box in covered),
            x1=max(box.x1 for box in covered),
            y1=max(box.y1 for box in covered),
        )

    @property
    def height(self) -> float:
        """How tall the extent is, in points."""
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        """How wide the extent is, in points."""
        return self.x1 - self.x0


class PageRegion(BaseModel):
    """A band of one page, cut for a model that reads pixels rather than text."""

    model_config = ConfigDict(frozen=True)

    page: int = Field(..., ge=1, description="1-based physical page the band is cut from.")
    clip: BBox = Field(..., description="Extent of the band on that page, in PDF points.")


class BodyProfile(BaseModel):
    """The document's own body style, the yardstick a run of text is judged against."""

    model_config = ConfigDict(frozen=True)

    size: float = Field(..., description="Modal (most common) body font size.")
    bold: bool = Field(default=False, description="Whether the body itself is mostly bold.")


class LayoutSpan(BaseModel):
    """One run of text with the style the document declares for it."""

    model_config = ConfigDict(frozen=True)

    BOLD: ClassVar[int] = 16
    """``flags`` bit set on bold characters."""

    ITALIC: ClassVar[int] = 2
    """``flags`` bit set on italic/oblique characters."""

    UNDERLINE: ClassVar[int] = 2
    """``char_flags`` bit set on underlined characters."""

    text: str
    size: float
    font: str
    flags: int = Field(0, description="Char style bitfield: bold and italic.")
    char_flags: int = Field(0, description="Character style bitfield: underline and strikeout.")
    source: str | None = Field(
        None,
        description=(
            "Where the run was read from, when the file names its own places: a cell "
            "reference such as C14. None for a run of glyphs, which has no name but its box."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bold(self) -> bool:
        """Whether the document renders this run bold."""
        return bool(self.flags & self.BOLD) or "bold" in self.font.lower()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def italic(self) -> bool:
        """Whether the document renders this run italic or oblique."""
        return bool(self.flags & self.ITALIC) or any(
            key in self.font.lower() for key in ("italic", "oblique")
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def underlined(self) -> bool:
        """Whether the document underlines this run."""
        return bool(self.char_flags & self.UNDERLINE)


class LayoutLine(BaseModel):
    """One physical line: its printable text, its extent, and the runs it is made of."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., description="Printable text of the line, whitespace collapsed.")
    bbox: BBox
    spans: list[LayoutSpan]

    turned: bool = Field(
        default=False,
        description="Whether the line is set at an angle rather than across the page.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size(self) -> float:
        """Largest run size on the line, in points."""
        return round(max(span.size for span in self.spans), 1) if self.spans else 0.0

    @property
    def furniture_key(self) -> str:
        """What a running header keeps across pages: its text minus the digits."""
        joined = "".join(c for c in TextSearch.simplified(self.text) if not c.isdigit())
        return " ".join(joined.split())


class TextBlock(BaseModel):
    """One text block of a page, as the document groups its lines."""

    model_config = ConfigDict(frozen=True)

    lines: list[LayoutLine] = Field(..., min_length=1)

    @property
    def text(self) -> str:
        """The block as it reads: its lines on one line, single-spaced."""
        return " ".join(line.text for line in self.lines)

    @property
    def bbox(self) -> BBox:
        """Extent of every line the block carries."""
        return BBox.union(line.bbox for line in self.lines) or self.lines[0].bbox


class PageSize(BaseModel):
    """The shape a page is printed at, in points."""

    model_config = ConfigDict(frozen=True)

    POINTS_PER_INCH: ClassVar[int] = 72
    """Points in an inch, the PDF unit a render resolution is scaled by."""

    width: float = Field(..., description="Page width in points, as the document declares it.")
    height: float = Field(..., description="Page height in points, as the document declares it.")


class Link(BaseModel):
    """One outbound link annotation and the extent it covers — over content, never a block."""

    model_config = ConfigDict(frozen=True)

    target: str = Field(
        ...,
        description="The annotation's destination as the document stores it: a URI for a URI "
        "action, a file path for a GoToR or Launch action.",
    )
    bbox: BBox


class RasterImage(BaseModel):
    """One raster placement on a page: an embedded pixel stream and the extent it covers."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["raster"] = "raster"
    bbox: BBox


class VectorDrawing(BaseModel):
    """One drawn region on a page — a plan, a chart, a drawn logo — which has no pixels."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["vector"] = "vector"
    bbox: BBox


# Everything a page paints beside text: sampled pixels, or a cluster of drawn paths.
PageImage = RasterImage | VectorDrawing


class LayoutPage(BaseModel):
    """One page: the shape it is printed at, its text blocks, and its running header/footer."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(..., ge=1, description="1-based physical page index.")
    size: PageSize = Field(..., description="The shape this page is printed at.")
    blocks: list[TextBlock] = Field(
        ...,
        description="The text blocks of the page, in reading order. Painted images live in "
        "``images``: a page that carries no text is told by ``lines`` being empty.",
    )
    links: list[Link] = Field(
        default=[],
        description="Every outbound link annotation of the page, in the document's own order. "
        "Internal jump links name no document and are not kept: the outline records them. "
        "``body_links`` reads this through the header/footer verdict.",
    )
    images: list[PageImage] = Field(
        default=[],
        description="Every image the page paints: raster placements first, vector drawings "
        "after. ``body_images`` reads this through the header/footer verdict.",
    )
    header_footer: frozenset[str] = Field(
        frozenset(),
        description="Normalized text of the running header/footer lines of this page, if any.",
    )

    @property
    def height(self) -> float:
        """How tall the page is, in points — the yardstick a position on it is read against."""
        return self.size.height

    @property
    def lines(self) -> list[LayoutLine]:
        """Every line of the page, in the document's own order."""
        return [line for block in self.blocks for line in block.lines]

    @property
    def text_is_extractable(self) -> bool:
        """Whether the parse rendered text for THIS page; a scanned page renders none."""
        return bool(self.lines)

    @property
    def furniture_lines(self) -> list[LayoutLine]:
        """The lines the verdict removes: the page's own edge lines, and no other."""
        lines = self.lines

        if not lines or not self.header_footer:
            return []

        ordered = sorted(lines, key=lambda line: (line.bbox.y0, line.text))
        return [
            line
            for line in {id(ordered[0]): ordered[0], id(ordered[-1]): ordered[-1]}.values()
            if line.furniture_key in self.header_footer
        ]

    @property
    def body_lines(self) -> list[LayoutLine]:
        """Every line of the page except its running header and footer."""
        furniture = {id(line) for line in self.furniture_lines}
        return [line for line in self.lines if id(line) not in furniture]

    @property
    def body_images(self) -> list[PageImage]:
        """The page's images minus header/footer furniture — the logos beside running lines."""
        header_bottom, footer_top = self._furniture_zone()
        return [
            image
            for image in self.images
            if not (image.bbox.y1 <= header_bottom or image.bbox.y0 >= footer_top)
        ]

    @property
    def body_links(self) -> list[Link]:
        """The page's links minus header/footer furniture — the site link under every page."""
        header_bottom, footer_top = self._furniture_zone()
        return [
            link
            for link in self.links
            if not (link.bbox.y1 <= header_bottom or link.bbox.y0 >= footer_top)
        ]

    @property
    def content_band(self) -> tuple[float, float]:
        """The y extent the page's own content occupies, its running header and footer cut off."""
        return self._furniture_zone()

    def _furniture_zone(self) -> tuple[float, float]:
        """Read the y extents the header/footer verdict proves on THIS page.

        Returns:
            (header_bottom, footer_top), content between the two; (0, height) with no verdict.
        """
        header_bottom, footer_top = 0.0, self.height
        middle = self.height / 2

        for line in self.furniture_lines:
            if line.bbox.y0 < middle:
                header_bottom = max(header_bottom, line.bbox.y1)
            else:
                footer_top = min(footer_top, line.bbox.y0)

        return header_bottom, footer_top

    @property
    def text(self) -> str:
        """The page as it reads: its body lines, one per line."""
        return "\n".join(line.text for line in self.body_lines)

    def with_header_footer(self, keys: frozenset[str]) -> "LayoutPage":
        """Return this page told which of its lines are running header/footer, not content.

        Args:
            keys: Normalized text of the lines that proved to repeat across neighbouring pages.

        Returns:
            The same page, carrying that verdict.
        """
        return self.model_copy(update={"header_footer": keys})


class OutlineEntry(BaseModel):
    """One bookmark of the embedded outline."""

    model_config = ConfigDict(frozen=True)

    title: str
    page: int | None = Field(
        None, description="1-based physical page; None when the bookmark resolves to none."
    )
    depth: int = Field(..., ge=1, description="1-based depth in the bookmark tree.")


class LayoutDocument(BaseModel):
    """A parsed document: its pages, their styled lines, and its embedded outline."""

    model_config = ConfigDict(frozen=True)

    origin: Literal["points", "cells"] = Field(
        default="points",
        description=(
            "What the coordinates mean. Points for a page, which is set in them; cells for a "
            "sheet, whose rows and columns ARE its geometry and which has no page to measure. "
            "Nothing that only compares one coordinate to another needs to read this — a "
            "section is sliced the same way in both — and only naming a place to a person does."
        ),
    )

    pages: list[LayoutPage]
    outline: list[OutlineEntry]
    checksum: str | None = Field(
        None,
        description="SHA-256 of the bytes this tree was parsed from; None when it was built "
        "from anything else.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def page_count(self) -> int:
        """Number of pages in the document."""
        return len(self.pages)

    @property
    def text(self) -> str:
        """Every printed line of the document, in its own order."""
        return "\n".join(page.text for page in self.pages)

    def body_profile(
        self, *, sample_pages: int, bold_ratio: float, default_size: float
    ) -> BodyProfile:
        """Measure the document's own body style: modal size and bold norm.

        Args:
            sample_pages: Divisor of the sampling stride (``page_count // this``), not a count.
            bold_ratio: Share of body characters that must be bold to call the body bold.
            default_size: Baseline size to fall back on when the document carries no text.

        Returns:
            The body size and whether the body is mostly bold.
        """
        sizes: Counter[float] = Counter()
        total = bold = 0
        step = max(1, self.page_count // sample_pages)

        for page in self.pages[::step]:
            # A running header or footer, often bold or large, would bias the yardstick.
            for line in page.body_lines:
                for span in line.spans:
                    width = len(span.text.strip())

                    if not width:
                        continue

                    sizes[round(span.size, 1)] += width
                    total += width

                    if span.bold:
                        bold += width

        size = sizes.most_common(1)[0][0] if sizes else default_size
        return BodyProfile(size=size, bold=total > 0 and bold / total >= bold_ratio)
