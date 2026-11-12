"""Unit tests for PageWindow: how a document is cut, and which sections each cut answers for."""

from typing import ClassVar

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    FormatCandidate,
    HeadingCandidates,
    OutlineCandidate,
    Typography,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.page_window import (
    PageWindow,
)
from baseline_core.core.llm.document_extraction.layout.layout import BBox
from baseline_core.core.llm.document_extraction.section import Section, SectionRelevance


class TestTile:
    """A document is cut into consecutive owned runs, each shown its own pages and no other."""

    STYLE: ClassVar[Typography] = Typography(size=14.0, bold=True)

    @classmethod
    def detected(cls, page: int) -> FormatCandidate:
        """Build one candidate on a page.

        Args:
            page: 1-based physical page it sits on.

        Returns:
            The candidate, its text naming its page.
        """
        return FormatCandidate(
            text=f"Heading {page}",
            typography=cls.STYLE,
            page=page,
            bbox=BBox(x0=72.0, y0=100.0, x1=400.0, y1=114.0),
            position=0.13,
        )

    @classmethod
    def spanning(cls, last_page: int) -> HeadingCandidates:
        """Build a document carrying one candidate on every page up to ``last_page``.

        Args:
            last_page: The page the last candidate sits on.

        Returns:
            The candidates of the whole document.
        """
        return HeadingCandidates(
            outline=[], format=[cls.detected(page) for page in range(1, last_page + 1)]
        )

    def test__a_document_shorter_than_the_window_is_one_whole_window(self) -> None:
        """Nothing is cut, and the window knows it stands for the whole document."""
        windows = PageWindow.tile(self.spanning(120), size=200)

        assert [(w.first_page, w.last_page, w.rank, w.count) for w in windows] == [(1, 120, 1, 1)]
        assert windows[0].partial is False
        assert windows[0].headings == self.spanning(120)

    def test__a_long_document_is_tiled_without_gaps_and_the_last_window_takes_the_rest(
        self,
    ) -> None:
        """The owned runs cover every page exactly once, however the length divides."""
        windows = PageWindow.tile(self.spanning(450), size=200)

        assert [(w.first_page, w.last_page) for w in windows] == [(1, 200), (201, 400), (401, 450)]
        assert [(w.rank, w.count) for w in windows] == [(1, 3), (2, 3), (3, 3)]
        assert all(w.partial for w in windows)

    def test__a_window_is_shown_its_own_pages_and_nothing_of_its_neighbours(self) -> None:
        """A page of the next window, even marked as context, tipped a whole window's levels."""
        middle = PageWindow.tile(self.spanning(450), size=200)[1]

        assert {c.page for c in middle.headings.format} == set(range(201, 401))

    def test__a_document_where_nothing_was_surfaced_yields_no_window(self) -> None:
        """No candidate means no call: there is no page to answer for."""
        assert PageWindow.tile(HeadingCandidates(outline=[], format=[]), size=200) == []

    def test__a_bookmark_beyond_the_last_candidate_still_extends_the_document(self) -> None:
        """Either source sets the last page, so a bookmarked tail page is never left unowned."""
        appendix = OutlineCandidate(text="Appendix", page=250)
        headings = HeadingCandidates(outline=[appendix], format=[self.detected(1)])

        windows = PageWindow.tile(headings, size=200)

        assert [(w.first_page, w.last_page) for w in windows] == [(1, 200), (201, 250)]
        assert windows[1].headings.outline == [appendix]


class TestOwns:
    """A window answers for its own run of pages and for nothing shown beside it."""

    WINDOW: ClassVar[PageWindow] = PageWindow(
        first_page=201,
        last_page=400,
        rank=2,
        count=3,
        headings=HeadingCandidates(outline=[], format=[]),
    )

    @staticmethod
    def section(page: int) -> Section:
        """Build a reported section opening on a page.

        Args:
            page: Its 1-based physical start page.

        Returns:
            The section.
        """
        return Section(title="x", start_page=page, level=1, relevance=SectionRelevance.IRRELEVANT)

    def test__a_section_on_an_owned_page_is_kept(self) -> None:
        """Both ends of the owned run are inclusive."""
        assert self.WINDOW.owns(self.section(201)) is True
        assert self.WINDOW.owns(self.section(300)) is True
        assert self.WINDOW.owns(self.section(400)) is True

    def test__a_section_filed_on_a_neighbours_page_is_the_neighbours(self) -> None:
        """A start page outside the run is reported by the window that owns that page."""
        assert self.WINDOW.owns(self.section(200)) is False
        assert self.WINDOW.owns(self.section(401)) is False

    def test__the_edge_windows_own_whatever_lies_beyond_the_document(self) -> None:
        """A page no window covers has no other owner, so a whole document keeps every answer."""
        first = self.WINDOW.model_copy(update={"rank": 1, "first_page": 1, "last_page": 200})
        last = self.WINDOW.model_copy(update={"rank": 3, "first_page": 401, "last_page": 450})

        assert first.owns(self.section(0)) is True
        assert first.owns(self.section(201)) is False
        assert last.owns(self.section(400)) is False
        assert last.owns(self.section(999)) is True
