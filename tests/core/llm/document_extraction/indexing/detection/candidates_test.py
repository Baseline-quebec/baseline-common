"""Unit tests for the detection-layer models that carry logic: candidates and serial codes."""

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    FormatCandidate,
    HeadingCandidates,
    OutlineCandidate,
    SerialCode,
    Typography,
)
from baseline_core.core.llm.document_extraction.layout.layout import BBox


class TestContinuation:
    """The marker that tells a reprinted rubric from a new one."""

    @staticmethod
    def continues(text: str) -> bool:
        """Read one line the way ``FormatCandidate.from_block`` reads it.

        Args:
            text: A candidate line, as printed.

        Returns:
            True when the line carries a continuation marker.
        """
        return FormatCandidate.CONTINUATION.search(text) is not None

    def test__english_and_french_markers_both_read_as_a_continuation(self) -> None:
        """A rubric reprinted to resume itself is marked in either language of a bilingual set."""
        assert self.continues("INSPECTION FINDINGS (continued)") is True
        assert self.continues("FINDINGS (cont'd)") is True
        assert self.continues("CONSTATS (suite)") is True
        assert self.continues("Annexe B (suite et fin)") is True

    def test__a_heading_without_the_marker_is_not_a_continuation(self) -> None:
        """The marker is the whole signal: nothing else promotes a heading to a continuation."""
        assert self.continues("FINDINGS OF THE INSPECTION") is False

    def test__the_word_alone_is_not_the_marker(self) -> None:
        """Suite is an ordinary word; only the parenthesised marker counts."""
        assert self.continues("SUITE OF FOLLOW-UP ACTIONS") is False


class TestHeadingCandidates:
    """What both sources surfaced, read by page."""

    @staticmethod
    def detected(page: int) -> FormatCandidate:
        """Build one candidate on a page.

        Args:
            page: 1-based physical page it sits on.

        Returns:
            The candidate.
        """
        return FormatCandidate(
            text=f"Heading {page}",
            typography=Typography(size=14.0, bold=True),
            page=page,
            bbox=BBox(x0=72.0, y0=100.0, x1=400.0, y1=114.0),
            position=0.13,
        )

    def test__nothing_from_either_source_is_empty(self) -> None:
        """One bookmark or one candidate is enough to make the document worth a call."""
        bookmark = OutlineCandidate(text="x", page=1)

        assert HeadingCandidates(outline=[], format=[]).empty is True
        assert HeadingCandidates(outline=[bookmark], format=[]).empty is False
        assert HeadingCandidates(outline=[], format=[self.detected(1)]).empty is False

    def test__the_last_page_is_the_farthest_either_source_reaches(self) -> None:
        """A bookmark past the last candidate still counts, and nothing at all is page 0."""
        headings = HeadingCandidates(
            outline=[OutlineCandidate(text="Appendix", page=40)],
            format=[self.detected(3), self.detected(12)],
        )

        assert headings.pages == {3, 12, 40}
        assert headings.last_page == 40
        assert HeadingCandidates(outline=[], format=[]).last_page == 0

    def test__between_keeps_both_sources_on_the_run_of_pages_inclusive(self) -> None:
        """Each source keeps its own order; a page outside the run drops from both."""
        headings = HeadingCandidates(
            outline=[OutlineCandidate(text="a", page=2), OutlineCandidate(text="b", page=9)],
            format=[self.detected(1), self.detected(5), self.detected(8), self.detected(9)],
        )

        kept = headings.between(2, 8)

        assert [e.page for e in kept.outline] == [2]
        assert [c.page for c in kept.format] == [5, 8]


class TestSerialCode:
    """The item code a line opens on, split into its family and its item number."""

    def test__a_code_splits_into_its_family_and_number(self) -> None:
        """The family is everything before the item number, the number read as an integer."""
        code = SerialCode.opening("RPT 26-12.02 Replacement of the pump")

        assert code is not None
        assert (code.code, code.family, code.number) == ("RPT 26-12.02", "RPT 26-12.", 2)

    def test__a_date_is_never_an_item_code(self) -> None:
        """A date counts like a series, so it is refused at the source."""
        assert SerialCode.opening("2025-12-04 site visit") is None
        assert SerialCode.opening("04/12/2025 site visit") is None

    def test__a_line_opening_on_a_word_carries_no_code(self) -> None:
        """A code cited mid-line opens nothing."""
        assert SerialCode.opening("As recorded under RPT 26-12.02") is None
