"""HeadingLocator — finds each reconciled heading back in the document and cuts its text out."""

import re
from math import inf
from operator import itemgetter
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.indexed_document import FileFormat
from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
)
from baseline_core.core.llm.document_extraction.section import (
    FileReference,
    LocatedSection,
    Section,
)
from baseline_core.core.llm.document_extraction.text_search import TextSearch


class Anchor(BaseModel):
    """Where one heading opens on its page, and how the locator came to believe it."""

    model_config = ConfigDict(frozen=True)

    line: int = Field(..., ge=0, description="Index of the printed line the heading opens on.")
    matched: bool = Field(
        ..., description="Whether the heading really cleared ``match_ratio`` on what was read."
    )
    extent: BBox = Field(..., description="Extent of the printed lines the heading was read on.")
    opener: bool = Field(
        default=False, description="Whether the anchor is the opener printed alone on its line."
    )


class PrintedLines(BaseModel):
    """A document's body lines laid end to end, and the slice of them each page owns."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    lines: list[LayoutLine] = Field(..., description="Every body line, in printed order.")
    spans: dict[int, range] = Field(..., description="The slice of ``lines`` each page owns.")
    pages: dict[int, LayoutPage] = Field(..., description="The document's pages, by number.")

    @classmethod
    def from_document(cls, document: LayoutDocument) -> Self:
        """Lay the document's printed lines end to end, remembering where each page starts.

        Args:
            document: The parsed document.

        Returns:
            Every body line in printed order, with the slice each page owns.
        """
        lines: list[LayoutLine] = []
        spans: dict[int, range] = {}

        for page in document.pages:
            body = page.body_lines
            spans[page.number] = range(len(lines), len(lines) + len(body))
            lines.extend(body)

        return cls(lines=lines, spans=spans, pages={page.number: page for page in document.pages})

    def page_of(self, index: int) -> int:
        """The page a printed line sits on.

        Args:
            index: The line's index in the flattened document.

        Returns:
            The 1-based number of the page owning that line.
        """
        return next(number for number, span in self.spans.items() if index in span)


class HeadingLocatorConfig(BaseModel):
    """Injectable knobs of the heading locator: its match threshold, the formats it links to."""

    model_config = ConfigDict(frozen=True)

    match_ratio: Annotated[
        int,
        Field(
            ge=0,
            le=100,
            description=(
                "Fuzzy score a window of printed lines must reach to be read as a section's "
                "heading. Scored whole, against the printed form — the opener the reconciler "
                "split off restored before the title — over a window sized to it."
            ),
        ),
    ] = 50
    document_formats: Annotated[
        list[FileFormat],
        Field(
            description=(
                "The formats a linked file must be in for the section to reference it — the "
                "ones the pipeline can take in. A link to a web page, or to any other format, "
                "is dropped at harvest."
            )
        ),
    ] = [FileFormat.PDF, FileFormat.XLSX]

    @property
    def document_suffixes(self) -> list[str]:
        """The file-name suffixes of every accepted format, in format order."""
        return [suffix for fmt in self.document_formats for suffix in fmt.suffixes]


class HeadingLocator:
    """Places each reconciled section back on the raw printed lines its heading opens on."""

    GLYPH_PAIRS: ClassVar[dict[str, str]] = {"999": "9", "WWW": "W"}
    """Two-line bullet glyphs an icon font prints as text; only the pair is dropped, never one."""

    GLYPH_RULE: ClassVar[re.Pattern[str]] = re.compile(r"^(\d)\1{7,}$")
    """A run of eight or more of one digit (« 88888888 »): an icon font's rule, never a number."""

    def __init__(self, config: HeadingLocatorConfig) -> None:
        """Bind the match threshold and the formats a section may reference.

        Args:
            config: The fuzzy-match threshold and the accepted document formats.
        """
        self._config = config

    def locate(self, sections: list[Section], document: LayoutDocument) -> list[LocatedSection]:
        """Place every reconciled section on its heading and give it the text under it.

        Args:
            sections: The sections the reconciler kept, in its own order.
            document: The parsed document those sections were read from.

        Returns:
            The same sections, in order, each carrying its heading extent and its text.
        """
        printed = PrintedLines.from_document(document)
        anchors = [self._anchor(section, printed) for section in sections]
        openings = [None if anchor is None else anchor.line for anchor in anchors]
        return [
            self._located(section, anchor, openings[rank + 1 :], printed)
            for rank, (section, anchor) in enumerate(zip(sections, anchors, strict=True))
        ]

    def _located(
        self,
        section: Section,
        anchor: Anchor | None,
        after: list[int | None],
        printed: PrintedLines,
    ) -> LocatedSection:
        """Give one section its heading extent, and the text and links down to the next section.

        Args:
            section: The reconciled section.
            anchor: Where its heading opens, or None when the document carries no such page.
            after: The opening lines of the sections following it, in reconciliation order.
            printed: Every printed line of the document, by page.

        Returns:
            The section, placed on its page, or bare when the page itself was not found.
        """
        if anchor is None:
            return LocatedSection(**section.model_dump())

        # Only a later line closes this section: one reconciled out of order is skipped.
        end = next(
            (later for later in after if later is not None and later > anchor.line),
            len(printed.lines),
        )
        return LocatedSection(
            **section.model_dump(),
            bbox=anchor.extent,
            text=self._joined(printed.lines[anchor.line : end]),
            matched=anchor.matched,
            opener_anchored=anchor.opener,
            file_references=self._references(anchor.line, end, printed),
        )

    def _references(self, start: int, end: int, printed: PrintedLines) -> list[FileReference]:
        """Harvest the documents linked in the band between a heading and the next one.

        Args:
            start: Index of the line the heading opens on.
            end: Index of the line the next section opens on, or one past the last line.
            printed: Every printed line of the document, by page.

        Returns:
            One reference per accepted document linked in that band, in printed order.
        """
        lines, pages = printed.lines, printed.pages
        opening = (printed.page_of(start), lines[start].bbox.y0)
        closing = (
            (printed.page_of(end), lines[end].bbox.y0) if end < len(lines) else (max(pages), inf)
        )
        # A link is placed by its centre: its box may start a point above the heading's top.
        links = [
            link
            for number in range(opening[0], closing[0] + 1)
            if number in pages
            for link in pages[number].body_links
            if opening <= (number, (link.bbox.y0 + link.bbox.y1) / 2) < closing
        ]
        return FileReference.from_links(links, self._config.document_suffixes)

    def _anchor(self, section: Section, printed: PrintedLines) -> Anchor | None:
        """Find the raw printed line a section's heading opens on, on its own page.

        Args:
            section: The section whose heading is looked up.
            printed: Every printed line of the document, by page.

        Returns:
            The anchor, or None when the document carries no such page, or that page no line.
        """
        span = printed.spans.get(section.start_page)

        if span is None or len(span) == 0:
            return None

        lines = printed.lines
        page = [lines[index].text for index in span]
        exact = self._exact_opener(section, page, lines, span)

        if exact is not None:
            return exact

        # Raw lines, never merged runs: a window sized to the heading reads a wrapped title whole.
        ranked = [
            (*TextSearch.fuzzy_window(section.printed_heading, page, start), start)
            for start in range(len(page))
        ]
        # Ties keep the first window: a heading is printed above the text that repeats it.
        score, stop, start = max(ranked, key=itemgetter(0))
        opening = span.start + start
        extent = (
            BBox.union(line.bbox for line in lines[opening : span.start + stop])
            or lines[opening].bbox
        )
        return Anchor(line=opening, matched=score >= self._config.match_ratio, extent=extent)

    @staticmethod
    def _exact_opener(
        section: Section, page: list[str], lines: list[LayoutLine], span: range
    ) -> Anchor | None:
        """Anchor on the opener wherever its page prints it alone on a line.

        Args:
            section: The section being anchored, carrying the opener when it has one.
            page: The text of the page's lines, in printed order.
            lines: Every printed line of the document.
            span: The slice of lines the section's page owns.

        Returns:
            A certain anchor, or None when the section has no opener or the page prints none.
        """
        if section.section_number is None:
            return None

        hits = TextSearch.exact_lines(section.section_number, page)

        if not hits:
            return None

        best = max(
            hits, key=lambda hit: TextSearch.fuzzy_window(section.printed_heading, page, hit)[0]
        )
        index = span.start + best
        return Anchor(line=index, matched=True, extent=lines[index].bbox, opener=True)

    @classmethod
    def _joined(cls, lines: list[LayoutLine]) -> str:
        """Read a run of printed lines as the text it makes up, icon-font glyphs dropped.

        Args:
            lines: The lines, in printed order.

        Returns:
            Their text, one line each, each bullet pair and digit rule dropped.
        """
        texts = [line.text for line in lines]
        kept: list[str] = []
        skip = False

        for index, text in enumerate(texts):
            if skip:
                skip = False
                continue

            stripped = text.strip()
            following = texts[index + 1].strip() if index + 1 < len(texts) else None

            if following is not None and cls.GLYPH_PAIRS.get(stripped) == following:
                skip = True
                continue

            if not cls.GLYPH_RULE.match(stripped):
                kept.append(text)

        return "\n".join(kept)
