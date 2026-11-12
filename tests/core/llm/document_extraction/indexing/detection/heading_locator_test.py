"""Unit tests for HeadingLocator: placing reconciled sections back on their heading line."""

from collections.abc import Callable
from typing import ClassVar

from baseline_core.core.llm.document_extraction.indexed_document import FileFormat
from baseline_core.core.llm.document_extraction.indexing.detection.heading_locator import (
    HeadingLocator,
    HeadingLocatorConfig,
)
from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutSpan,
    Link,
    PageSize,
    TextBlock,
)
from baseline_core.core.llm.document_extraction.section import Section, SectionRelevance


class PageBuilder:
    """Builds one-page documents of printed lines, each carrying a real span."""

    LETTER: ClassVar[PageSize] = PageSize(width=612.0, height=792.0)

    @staticmethod
    def line(text: str, y0: float, make_span: Callable[..., LayoutSpan]) -> LayoutLine:
        """Build one printed line at a height, carrying a real span for the body profile.

        Args:
            text: The line, as printed.
            y0: Its top edge, in points.
            make_span: The span factory.

        Returns:
            The line.
        """
        return LayoutLine(
            text=text, bbox=BBox(x0=72.0, y0=y0, x1=540.0, y1=y0 + 12.0), spans=[make_span(text)]
        )

    @classmethod
    def document(
        cls, rows: list[list[tuple[str, float]]], make_span: Callable[..., LayoutSpan]
    ) -> LayoutDocument:
        """Build a one-page document, one block per row group.

        Args:
            rows: Each block's lines, as (text, y0).
            make_span: The span factory.

        Returns:
            The document.
        """
        blocks = [
            TextBlock(lines=[cls.line(text, y0, make_span) for text, y0 in block])
            for block in rows
        ]
        return LayoutDocument(
            pages=[LayoutPage(number=1, size=cls.LETTER, blocks=blocks)], outline=[]
        )

    @staticmethod
    def section(title: str, start_page: int = 1, section_number: str | None = None) -> Section:
        """Build an uncertain section the way the reconciler would emit it.

        Args:
            title: The heading description, as the reconciler kept it.
            start_page: The 1-based physical page it claims.
            section_number: The opener the title was split from, if any.

        Returns:
            The section, at the top level and left uncertain.
        """
        return Section(
            title=title,
            start_page=start_page,
            level=1,
            relevance=SectionRelevance.UNCERTAIN,
            section_number=section_number,
        )


class TestHeadingLocator:
    def test__places_each_section_on_the_line_its_heading_is_printed_on(
        self, heading_locator: HeadingLocator, with_outline_layout: LayoutDocument
    ) -> None:
        """Every reconciled title is found again, and its text opens on that heading."""
        sections = [PageBuilder.section("Introduction", 2), PageBuilder.section("Methods", 3)]

        located = heading_locator.locate(sections, with_outline_layout)

        assert [s.matched for s in located] == [True, True]
        assert [s.text.startswith(s.title) for s in located] == [True, True]

    def test__a_sections_text_stops_at_the_next_located_section(
        self, heading_locator: HeadingLocator, with_outline_layout: LayoutDocument
    ) -> None:
        """The next heading closes the previous section rather than joining it."""
        sections = [PageBuilder.section("Introduction", 2), PageBuilder.section("Methods", 3)]

        intro = heading_locator.locate(sections, with_outline_layout)[0]

        assert "explanatory prose" in intro.text
        assert "Methods" not in intro.text

    def test__the_last_section_runs_to_the_end_of_the_document(
        self, heading_locator: HeadingLocator, with_outline_layout: LayoutDocument
    ) -> None:
        """Nothing closes the last section, so it keeps every line under it."""
        methods = heading_locator.locate([PageBuilder.section("Methods", 3)], with_outline_layout)

        assert "Results" in methods[0].text

    def test__a_title_no_line_carries_falls_back_to_the_best_line_of_its_page(
        self, heading_locator: HeadingLocator, with_outline_layout: LayoutDocument
    ) -> None:
        """A title the page does not print still gets that page's best line, flagged unmatched."""
        section = PageBuilder.section("Civil safety elements", 2)

        guessed = heading_locator.locate([section], with_outline_layout)[0]

        assert guessed.matched is False
        assert guessed.bbox is not None
        assert guessed.text

    def test__a_section_pointing_at_no_page_of_the_document_is_left_unplaced(
        self, heading_locator: HeadingLocator, with_outline_layout: LayoutDocument
    ) -> None:
        """A start page the document does not have carries neither a position nor text."""
        section = PageBuilder.section("Introduction", 99)

        unplaced = heading_locator.locate([section], with_outline_layout)[0]

        assert unplaced.bbox is None
        assert unplaced.matched is False
        assert not unplaced.text

    def test__icon_font_bullets_and_rules_never_reach_the_text(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """A « 999 » / « 9 » bullet pair and a digit rule go; a lone « 999 » stays."""
        document = PageBuilder.document(
            [
                [("Pump room", 100.0)],
                [("999", 130.0), ("9", 145.0), ("Replace the seals", 160.0)],
                [("88888888", 190.0), ("999", 220.0)],
            ],
            make_span,
        )

        located = heading_locator.locate([PageBuilder.section("Pump room")], document)[0]

        assert located.text.splitlines() == ["Pump room", "Replace the seals", "999"]


class TestOpenerAnchorsTheMatch:
    """Template titles repeat: only the opener says which printing is which."""

    TITLE: ClassVar[str] = "NOTICE AND SUBMISSION OF THE DRAFT PROCEDURE"

    def document(self, make_span: Callable[..., LayoutSpan]) -> LayoutDocument:
        """One page printing the same template title twice, under two openers.

        Args:
            make_span: The span factory.

        Returns:
            The document.
        """
        return PageBuilder.document(
            [
                [(f"8.1 {self.TITLE}", 100.0)],
                [("The inspector takes note of the submission.", 130.0)],
                [(f"8.2 {self.TITLE}", 300.0)],
                [("The recommendation is approved.", 330.0)],
            ],
            make_span,
        )

    def test__the_opener_tells_two_template_titles_apart(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """The second printing of a template title is found by the opener it opens on."""
        section = PageBuilder.section(self.TITLE, section_number="8.2")

        located = heading_locator.locate([section], self.document(make_span))[0]

        assert located.matched is True
        assert located.text.startswith(f"8.2 {self.TITLE}")

    def test__a_section_without_an_opener_still_anchors_on_its_title(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """Restoring nothing changes nothing: a bare title matches as it always did."""
        section = PageBuilder.section(self.TITLE)

        located = heading_locator.locate([section], self.document(make_span))[0]

        assert located.matched is True
        assert self.TITLE in located.text


class TestHeadingsSplitOverLines:
    """An opener filed on its own line over the title: both lines are the heading."""

    TITLE: ClassVar[str] = "Submission of the report of the technical review committee"
    OPENER: ClassVar[str] = "TR-260120-2.1"
    GARBLED: ClassVar[str] = "A title the model transcribed quite differently"

    def locate(
        self,
        heading_locator: HeadingLocator,
        make_span: Callable[..., LayoutSpan],
        rows: list[list[tuple[str, float]]],
    ) -> str:
        """Locate the opener-carrying section on one page of ``rows`` and return its text.

        Args:
            heading_locator: The locator under test.
            make_span: The span factory.
            rows: Each block's lines, as (text, y0).

        Returns:
            The located section's text.
        """
        section = PageBuilder.section(self.TITLE, section_number=self.OPENER)

        located = heading_locator.locate([section], PageBuilder.document(rows, make_span))[0]

        assert located.matched is True
        return located.text

    def test__an_opener_on_its_own_line_opens_the_section(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """The code line above the title is the section's own, never the previous one's tail."""
        rows = [
            [("The previous section ends here.", 80.0)],
            [(self.OPENER, 110.0), (self.TITLE, 124.0)],
            [("The inspector takes note of the submission.", 150.0)],
        ]

        assert self.locate(heading_locator, make_span, rows).startswith(self.OPENER)

    def test__a_block_boundary_between_opener_and_title_changes_nothing(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """The locator reads raw lines: how the parser blocked them must not move the anchor."""
        rows = [
            [("The previous section ends here.", 80.0)],
            [(self.OPENER, 110.0)],
            [(self.TITLE, 140.0)],
            [("The inspector takes note of the submission.", 170.0)],
        ]

        assert self.locate(heading_locator, make_span, rows).startswith(self.OPENER)

    def test__an_opener_over_a_wrapped_title_opens_the_section(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """Three printed lines, one heading: the window covers them all."""
        rows = [
            [("The previous section ends here.", 80.0)],
            [(self.OPENER, 110.0)],
            [("Submission of the report of the technical", 140.0), ("review committee", 154.0)],
            [("The inspector takes note of the submission.", 180.0)],
        ]

        assert self.locate(heading_locator, make_span, rows).startswith(self.OPENER)

    def test__a_unique_opener_line_anchors_with_certainty(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """An exact opener needs no score: even a garbled title anchors on its code line."""
        document = PageBuilder.document(
            [
                [("The previous section ends here.", 80.0)],
                [(self.OPENER, 110.0)],
                [("A heading the page prints in other words", 140.0)],
            ],
            make_span,
        )
        section = PageBuilder.section(self.GARBLED, section_number=self.OPENER)

        located = heading_locator.locate([section], document)[0]

        assert located.matched is True
        assert located.text.startswith(self.OPENER)

    def test__a_repeated_opener_anchors_on_the_printing_its_title_sits_under(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """Two identical opener lines settle nothing alone: the title under each decides which."""
        document = PageBuilder.document(
            [
                [("2.1", 80.0)],
                [("Another rubric of the same rank", 100.0)],
                [("2.1", 300.0)],
                [(self.TITLE, 320.0)],
            ],
            make_span,
        )
        section = PageBuilder.section(self.TITLE, section_number="2.1")

        located = heading_locator.locate([section], document)[0]

        assert located.matched is True
        # The window reads the SECOND « 2.1 » with the title under it, opener line included.
        assert located.text.split("\n")[:2] == ["2.1", self.TITLE]

    def test__a_repeated_opener_still_anchors_on_an_opener_when_the_title_is_garbled(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """A title the page never prints cannot pull the anchor off the lines the opener names."""
        document = PageBuilder.document(
            [
                [("An unrelated rubric", 60.0)],
                [("2.1", 80.0)],
                [("Another rubric of the same rank", 100.0)],
                [("2.1", 300.0)],
            ],
            make_span,
        )
        section = PageBuilder.section(self.GARBLED, section_number="2.1")

        located = heading_locator.locate([section], document)[0]

        assert located.matched is True
        assert located.text.startswith("2.1")

    def test__an_opener_anchor_is_remembered_as_one(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """A section anchored on its opener bar says so; a title-matched one claims nothing."""
        document = PageBuilder.document([[(self.OPENER, 110.0)], [(self.TITLE, 140.0)]], make_span)
        numbered = PageBuilder.section(self.TITLE, section_number=self.OPENER)
        bare = PageBuilder.section(self.TITLE)

        on_opener, on_title = heading_locator.locate([numbered, bare], document)

        assert on_opener.opener_anchored is True
        assert on_title.opener_anchored is False


class TestLinksUnderTheHeading:
    """The links printed between a heading and the next one are the section's references."""

    @staticmethod
    def link(target: str, y0: float) -> Link:
        """Build one link annotation boxed at a height, as tall as a printed line.

        Args:
            target: Where the link points.
            y0: Its top edge, in points.

        Returns:
            The link.
        """
        return Link(target=target, bbox=BBox(x0=72.0, y0=y0, x1=150.0, y1=y0 + 12.0))

    def document(self, make_span: Callable[..., LayoutSpan]) -> LayoutDocument:
        """Two pages: two decisions on the first, the second one running onto the next.

        Args:
            make_span: The span factory.

        Returns:
            The document.
        """
        first = PageBuilder.document(
            [
                [("CV-2025-1367", 118.0)],
                [("Partial release of an easement", 118.0)],
                [("On the proposal of the inspector.", 300.0)],
                [("Creation of an operating surplus", 500.0)],
                [("The surplus is created as proposed.", 600.0)],
            ],
            make_span,
        ).pages[0]
        second = PageBuilder.document([[("Approved as proposed.", 100.0)]], make_span).pages[0]
        return LayoutDocument(
            pages=[
                first.model_copy(
                    update={
                        "links": [
                            # The margin code's box starts a point above its heading's top.
                            self.link("https://example.org/CV-2025-1367.pdf", 117.0),
                            self.link("https://example.org/DE2025-445.pdf", 300.0),
                            self.link("https://example.org/safety/", 320.0),
                            self.link("https://example.org/CV-2025-1368.pdf", 499.0),
                        ]
                    }
                ),
                second.model_copy(
                    update={
                        "number": 2,
                        "links": [self.link("https://example.org/FN2025-066.xlsx", 100.0)],
                    }
                ),
            ],
            outline=[],
        )

    def test__each_link_goes_to_the_section_whose_band_holds_it(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """A margin code level with its heading stays with it; the next heading closes the band."""
        sections = [
            PageBuilder.section("Partial release of an easement"),
            PageBuilder.section("Creation of an operating surplus"),
        ]

        first, second = heading_locator.locate(sections, self.document(make_span))

        assert [r.filename for r in first.file_references] == [
            "CV-2025-1367.pdf",
            "DE2025-445.pdf",
        ]
        assert [r.filename for r in second.file_references] == [
            "CV-2025-1368.pdf",
            "FN2025-066.xlsx",
        ]

    def test__a_web_page_is_never_a_reference(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """The site link printed inside the band names no document."""
        section = PageBuilder.section("Partial release of an easement")

        located = heading_locator.locate([section], self.document(make_span))[0]

        assert all("safety" not in r.target for r in located.file_references)

    def test__the_accepted_formats_are_the_configs(
        self, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """Configured for PDF alone, the locator drops the workbook the last band links to."""
        locator = HeadingLocator(HeadingLocatorConfig(document_formats=[FileFormat.PDF]))
        section = PageBuilder.section("Creation of an operating surplus")

        located = locator.locate([section], self.document(make_span))[0]

        assert [r.filename for r in located.file_references] == ["CV-2025-1368.pdf"]

    def test__an_unplaced_section_references_nothing(
        self, heading_locator: HeadingLocator, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """No heading found means no band to harvest."""
        section = PageBuilder.section("Partial release of an easement", 99)

        located = heading_locator.locate([section], self.document(make_span))[0]

        assert located.file_references == []
