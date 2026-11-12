"""Unit tests for the vision router: every kept section cut out of the pages it runs across."""

from itertools import starmap
from typing import ClassVar

from baseline_core.core.llm.document_extraction.extraction.vision_router import VisionRouter
from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutSpan,
    PageSize,
    TextBlock,
)
from baseline_core.core.llm.document_extraction.section import LocatedSection, SectionRelevance


class TestVisionRouter:
    """A section is cut from its own heading down to the next one, header and footer dropped."""

    SIZE: ClassVar[PageSize] = PageSize(width=80.0, height=100.0)
    SHEET: ClassVar[PageSize] = PageSize(width=200.0, height=100.0)

    @staticmethod
    def line(text: str, y0: float) -> LayoutLine:
        """One printed line at a height, carrying the span the body profile is read from."""
        return LayoutLine(
            text=text,
            bbox=BBox(x0=10, y0=y0, x1=90, y1=y0 + 10),
            spans=[LayoutSpan(text=text, size=11.0, font="Helvetica")],
        )

    def page(
        self,
        number: int,
        *texts: tuple[str, float],
        footer: str = "",
        size: PageSize | None = None,
    ) -> LayoutPage:
        """One page printing each text at its height, with an optional running footer."""
        lines = list(starmap(self.line, texts))

        if footer:
            lines.append(self.line(footer, 95.0))

        return LayoutPage(
            number=number,
            size=size or self.SIZE,
            blocks=[TextBlock(lines=lines)],
            header_footer=frozenset({self.line(footer, 95.0).furniture_key})
            if footer
            else frozenset(),
        )

    def section(
        self, start: int, y0: float, relevance: SectionRelevance, *, opener: bool = False
    ) -> LocatedSection:
        """A located section opening at a height on a page."""
        return LocatedSection(
            title=f"s{start}-{y0:.0f}",
            start_page=start,
            level=1,
            relevance=relevance,
            bbox=BBox(x0=10, y0=y0, x1=90, y1=y0 + 10),
            opener_anchored=opener,
        )

    def test__a_section_is_cut_from_its_heading_to_the_next_one(self) -> None:
        """Two sections on one page each get their own band, split where the second opens."""
        document = LayoutDocument(pages=[self.page(1, ("One", 10.0), ("Two", 50.0))], outline=[])
        sections = [
            self.section(1, 10.0, SectionRelevance.RELEVANT),
            self.section(1, 50.0, SectionRelevance.RELEVANT),
        ]

        first, second = VisionRouter.route(sections, document)

        assert [(r.page, r.clip.y0, r.clip.y1) for r in first] == [(1, 10.0, 50.0)]
        assert [(r.page, r.clip.y0, r.clip.y1) for r in second] == [(1, 50.0, 100.0)]

    def test__a_section_spanning_pages_is_cut_once_per_page(self) -> None:
        """A section running across three pages yields one band per page it crosses."""
        document = LayoutDocument(
            pages=[
                self.page(1, ("One", 60.0)),
                self.page(2, ("Continued", 10.0)),
                self.page(3, ("Two", 40.0)),
            ],
            outline=[],
        )
        sections = [
            self.section(1, 60.0, SectionRelevance.RELEVANT),
            self.section(3, 40.0, SectionRelevance.RELEVANT),
        ]

        first = VisionRouter.route(sections, document)[0]

        assert [(r.page, r.clip.y0, r.clip.y1) for r in first] == [
            (1, 60.0, 100.0),
            (2, 0.0, 100.0),
            (3, 0.0, 40.0),
        ]

    def test__the_running_footer_is_left_out_of_the_band(self) -> None:
        """A page's own footer is furniture: the band stops above it."""
        document = LayoutDocument(
            pages=[
                self.page(1, ("One", 10.0), footer="Page 1 of 2"),
                self.page(2, ("Two", 10.0), footer="Page 1 of 2"),
            ],
            outline=[],
        )
        sections = [self.section(1, 10.0, SectionRelevance.RELEVANT)]

        bands = VisionRouter.route(sections, document)[0]

        assert [r.clip.y1 for r in bands] == [95.0, 95.0]

    def test__a_sheet_is_cut_out_of_nothing(self) -> None:
        """A workbook paints no pixels, so no section of it gets a band."""
        document = LayoutDocument(origin="cells", pages=[self.page(1, ("One", 10.0))], outline=[])
        sections = [self.section(1, 10.0, SectionRelevance.RELEVANT)]

        assert VisionRouter.route(sections, document) == [[]]

    def test__an_irrelevant_section_is_cut_out_of_nothing(self) -> None:
        """Nothing downstream reads an irrelevant section, so its pixels are never rendered."""
        document = LayoutDocument(pages=[self.page(1, ("One", 10.0))], outline=[])
        sections = [self.section(1, 10.0, SectionRelevance.IRRELEVANT)]

        assert VisionRouter.route(sections, document) == [[]]

    def test__an_uncertain_section_is_cut_like_a_relevant_one(self) -> None:
        """Only « irrelevant » is terminal: an uncertain section still reaches the eye."""
        document = LayoutDocument(pages=[self.page(1, ("One", 10.0))], outline=[])
        sections = [self.section(1, 10.0, SectionRelevance.UNCERTAIN)]

        assert len(VisionRouter.route(sections, document)[0]) == 1

    def test__a_section_the_locator_placed_nowhere_is_cut_out_of_nothing(self) -> None:
        """No position, no band: a section carrying no bbox is rendered from nothing."""
        document = LayoutDocument(pages=[self.page(1, ("One", 10.0))], outline=[])
        unplaced = LocatedSection(
            title="nowhere", start_page=1, level=1, relevance=SectionRelevance.RELEVANT
        )

        assert VisionRouter.route([unplaced], document) == [[]]

    def test__a_bar_anchored_section_opens_its_page(self) -> None:
        """What a form page prints above its opener bar is form header, never the last tail."""
        document = LayoutDocument(
            pages=[
                self.page(1, ("One", 10.0)),
                self.page(2, ("Form letterhead", 10.0), ("Item # : 1", 50.0)),
            ],
            outline=[],
        )
        sections = [
            self.section(1, 10.0, SectionRelevance.RELEVANT),
            self.section(2, 50.0, SectionRelevance.RELEVANT, opener=True),
        ]

        first, second = VisionRouter.route(sections, document)

        assert [(r.page, r.clip.y0, r.clip.y1) for r in first] == [(1, 10.0, 100.0)]
        assert [(r.page, r.clip.y0, r.clip.y1) for r in second] == [(2, 50.0, 100.0)]

    def test__a_drawing_sheet_is_cut_whole(self) -> None:
        """A sheet caption owns the drawing above it: the band is the page, never a strip."""
        document = LayoutDocument(
            pages=[
                self.page(1, ("One", 10.0)),
                self.page(2, ("NORTH ELEVATION", 80.0), size=self.SHEET),
            ],
            outline=[],
        )
        sections = [
            self.section(1, 10.0, SectionRelevance.RELEVANT),
            self.section(2, 80.0, SectionRelevance.RELEVANT),
        ]

        first, second = VisionRouter.route(sections, document)

        assert [(r.page, r.clip.y0, r.clip.y1) for r in first] == [(1, 10.0, 100.0)]
        assert [(r.page, r.clip.y0, r.clip.y1) for r in second] == [(2, 0.0, 100.0)]

    def test__two_captions_on_one_sheet_each_take_it_whole(self) -> None:
        """Side-by-side panels split nothing: each panel section reads its whole sheet."""
        document = LayoutDocument(
            pages=[self.page(1, ("DETAIL A", 40.0), ("DETAIL B", 40.0), size=self.SHEET)],
            outline=[],
        )
        sections = [
            self.section(1, 40.0, SectionRelevance.RELEVANT),
            self.section(1, 40.0, SectionRelevance.RELEVANT),
        ]

        first, second = VisionRouter.route(sections, document)

        assert [(r.page, r.clip.y0, r.clip.y1) for r in first] == [(1, 0.0, 100.0)]
        assert [(r.page, r.clip.y0, r.clip.y1) for r in second] == [(1, 0.0, 100.0)]
