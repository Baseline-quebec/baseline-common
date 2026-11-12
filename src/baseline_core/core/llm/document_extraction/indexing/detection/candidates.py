"""Data models of the detection layer: styles, positions, candidates, serial item codes."""

import re
import string
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.layout import BBox


class Typography(BaseModel):
    """How a run is set: its size and the emphasis carried over it."""

    model_config = ConfigDict(frozen=True)

    size: float
    bold: bool = False
    italic: bool = False
    underlined: bool = False
    caps: bool = False


class StyledText(BaseModel):
    """A run of text and the typography it is set in — one physical line or a merged chunk."""

    model_config = ConfigDict(frozen=True)

    text: str
    typography: Typography


class StyledBlock(StyledText):
    """Contiguous lines merged into one unit, and where on its page that unit sits."""

    page: Annotated[int, Field(ge=1, description="1-based physical page the chunk sits on.")]
    bbox: Annotated[BBox, Field(description="Extent of the merged lines, always measured.")]


class FormatCandidate(StyledBlock):
    """A styled run the typography surfaced — read off the page, so always located on it."""

    CONTINUATION: ClassVar[re.Pattern[str]] = re.compile(
        r"\(\s*(?:suite(?:\s+et\s+fin)?|continued|cont'?d)\s*\)", re.IGNORECASE
    )
    """The marker a heading carries when it resumes a section that started further up."""

    position: Annotated[
        float,
        Field(description="How far down its page the run sits, in shares of the height, 0 = top."),
    ]
    continues: Annotated[
        bool,
        Field(description="Whether the line carries a '(continued)' marker resuming its section."),
    ] = False

    @classmethod
    def from_block(cls, block: StyledBlock, page_height: float) -> Self:
        """Promote a merged chunk by measuring how far down its own page it starts.

        Args:
            block: The chunk the finder merged.
            page_height: Height of that chunk's page, in points.

        Returns:
            The candidate, carrying the chunk's text, style, page and extent.
        """
        return cls(
            text=block.text,
            typography=block.typography,
            page=block.page,
            bbox=block.bbox,
            position=block.bbox.y0 / page_height,
            continues=cls.CONTINUATION.search(block.text) is not None,
        )


class OutlineCandidate(BaseModel):
    """A bookmark of the embedded outline: a title and the page it points at."""

    model_config = ConfigDict(frozen=True)

    text: str
    page: Annotated[int, Field(ge=1, description="1-based physical page the bookmark points at.")]


class HeadingCandidates(BaseModel):
    """Everything the finder surfaced from one document, source by source."""

    model_config = ConfigDict(frozen=True)

    outline: list[OutlineCandidate]
    format: list[FormatCandidate]

    @property
    def empty(self) -> bool:
        """Whether neither source surfaced anything."""
        return not self.outline and not self.format

    @property
    def pages(self) -> set[int]:
        """Every page either source lands on."""
        return {entry.page for entry in self.outline} | {heading.page for heading in self.format}

    @property
    def last_page(self) -> int:
        """The last page either source lands on; 0 when nothing was surfaced."""
        return max(self.pages, default=0)

    def between(self, first: int, last: int) -> "HeadingCandidates":
        """Keep what both sources surfaced on a run of pages.

        Args:
            first: The first page kept.
            last: The last page kept, inclusive.

        Returns:
            The entries landing on those pages, each source in its own order.
        """
        return HeadingCandidates(
            outline=[entry for entry in self.outline if first <= entry.page <= last],
            format=[heading for heading in self.format if first <= heading.page <= last],
        )


class SerialCode(BaseModel):
    """A line-opening item code of a numbered series, split into its family and item number."""

    model_config = ConfigDict(frozen=True)

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*(?P<code>[A-Za-z]{0,6}[ \-]?\d{1,6}(?:[\-./]\d{1,6}){1,3})(?=\s|$)"
    )
    """A line-opening item code of two to four numeric groups (« BT-2026-0145 », « SH-2.1 »)."""

    DATE: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4})$"
    )
    """A date opening a line — numbered like a series, never an item."""

    code: str = Field(..., description="The code, single-spaced, so two printings compare equal.")
    family: str = Field(..., description="Everything before the item number (« BT-2026- »).")
    number: int = Field(..., description="The item number the code ends on (« 0145 » -> 145).")

    @classmethod
    def opening(cls, text: str) -> Self | None:
        """Read the item code a line opens on.

        Args:
            text: A printed line or a candidate, as printed.

        Returns:
            The code, or None when the line opens on no code, or on a date.
        """
        match = cls.PATTERN.match(text)

        if match is None or cls.DATE.match(match["code"]):
            return None

        code = " ".join(match["code"].split())
        family = code.rstrip(string.digits)
        return cls(code=code, family=family, number=int(code[len(family) :]))


class SerialItem(BaseModel):
    """One item of a counting series the finder promoted: its code and its heading candidate."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(..., description="The item's code, single-spaced.")
    candidate: FormatCandidate = Field(..., description="The candidate the item opens on.")
