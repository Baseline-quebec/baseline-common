"""Unit tests locking what the reconciler's prompts show the model."""

from typing import ClassVar

import pytest

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    FormatCandidate,
    HeadingCandidates,
    OutlineCandidate,
    Typography,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.config import (
    SectionReconcilerConfig,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.page_window import (
    PageWindow,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.prompt import (
    SectionReconcilerPrompt,
)
from baseline_core.core.llm.document_extraction.layout.layout import BBox


class TestSystem:
    """The system prompt is the library's, with the project's criterion written in."""

    def test__the_criterion_is_written_into_the_relevance_rules(
        self, reconciler_config: SectionReconcilerConfig
    ) -> None:
        """The project states relevance; the rules around it stay the library's."""
        system = SectionReconcilerPrompt(reconciler_config).system

        criterion = system.index(reconciler_config.relevance_criterion)
        assert system.index("<relevance_rules>\n") < criterion < system.index("</relevance_rules>")
        assert SectionReconcilerPrompt.RELEVANCE_SLOT not in system

    def test__no_window_rule_lives_in_the_system_prompt(
        self, reconciler_config: SectionReconcilerConfig
    ) -> None:
        """A rule about windows in the instructions pushed whole documents a level down."""
        assert "Window" not in SectionReconcilerPrompt(reconciler_config).system


class TestUser:
    """The candidates are the ground truth; the outline metadata only points at pages."""

    PAGE_HEIGHT: ClassVar[float] = 792.0
    """Height of the letter page every candidate of these tests is read from."""

    TITLE: ClassVar[Typography] = Typography(size=14.0, bold=True)
    """The typography every format candidate of these tests is set in."""

    RUBRIC: ClassVar[Typography] = Typography(size=14.0, bold=True, caps=True)
    """The same typography, on a line the finder read as printed in capitals."""

    @classmethod
    def detected(cls, text: str, page: int, y0: float) -> FormatCandidate:
        """Build a format candidate whose top edge sits at ``y0`` on a letter page.

        Args:
            text: The detected line, as printed.
            page: 1-based physical page it sits on.
            y0: Top edge, in points from the page top.

        Returns:
            The candidate the renderer reads, set in ``TITLE``.
        """
        return FormatCandidate(
            text=text,
            typography=cls.TITLE,
            page=page,
            bbox=BBox(x0=72.0, y0=y0, x1=400.0, y1=y0 + 14.0),
            position=y0 / cls.PAGE_HEIGHT,
        )

    @pytest.fixture
    def prompt(self, reconciler_config: SectionReconcilerConfig) -> SectionReconcilerPrompt:
        """Return the prompt at the library defaults."""
        return SectionReconcilerPrompt(reconciler_config)

    @staticmethod
    def render(prompt: SectionReconcilerPrompt, headings: HeadingCandidates) -> str:
        """Render a short document's user message.

        Args:
            prompt: The prompt under test.
            headings: Both detection sources of the document.

        Returns:
            The user message.
        """
        return prompt.user(PageWindow.tile(headings, 400)[0], None)

    def test__a_candidate_renders_verbatim_with_its_printing(
        self, prompt: SectionReconcilerPrompt
    ) -> None:
        """The bracket opens the bullet with the size, the emphasis and the vertical position."""
        headings = HeadingCandidates(outline=[], format=[self.detected("INTRODUCTION", 1, 79.2)])

        rendered = self.render(prompt, headings)

        assert '- [14pt bold, 10% down the page]  "INTRODUCTION"' in rendered

    def test__a_capitalized_candidate_carries_the_caps_mark(
        self, prompt: SectionReconcilerPrompt
    ) -> None:
        """Where every heading is set alike, the mark is the only rank the model is given."""
        headings = HeadingCandidates(
            outline=[],
            format=[
                self.detected("FINANCES", 1, 79.2).model_copy(update={"typography": self.RUBRIC})
            ],
        )

        rendered = self.render(prompt, headings)

        assert '- [14pt bold CAPS, 10% down the page]  "FINANCES"' in rendered

    def test__a_resuming_candidate_closes_its_bracket_on_the_continuation_mark(
        self, prompt: SectionReconcilerPrompt
    ) -> None:
        """The model must be told a line resumes a section, not left to spot the marker."""
        headings = HeadingCandidates(
            outline=[],
            format=[
                self.detected("FINDINGS (continued)", 2, 79.2).model_copy(
                    update={"continues": True}
                )
            ],
        )

        rendered = self.render(prompt, headings)

        assert '- [14pt bold, 10% down the page, continuation]  "FINDINGS (continued)"' in rendered

    def test__a_contents_line_loses_its_dot_leaders_and_printed_page(
        self, prompt: SectionReconcilerPrompt
    ) -> None:
        """Only the dot-leader form goes: the filler and the page it points at."""
        headings = HeadingCandidates(
            outline=[], format=[self.detected("1. Introduction ........ 5", 1, 79.2)]
        )

        rendered = self.render(prompt, headings)

        assert ']  "1. Introduction"' in rendered

    def test__candidates_render_in_page_then_vertical_order(
        self, prompt: SectionReconcilerPrompt
    ) -> None:
        """A later-page candidate never precedes an earlier one; same page sorts by y0."""
        headings = HeadingCandidates(
            outline=[],
            format=[
                self.detected("OPEN QUESTIONS", 1, 300.0),
                self.detected("SCOPE OF THE REVIEW", 1, 100.0),
                self.detected("SIGN-OFF", 2, 100.0),
            ],
        )

        rendered = self.render(prompt, headings)

        assert (
            rendered.index("SCOPE OF THE REVIEW")
            < rendered.index("OPEN QUESTIONS")
            < rendered.index("SIGN-OFF")
        )

    def test__a_bookmark_renders_under_its_own_page_before_the_candidates(
        self, prompt: SectionReconcilerPrompt
    ) -> None:
        """The metadata leads the page block it points at, and never fuses with a candidate."""
        headings = HeadingCandidates(
            outline=[OutlineCandidate(text="20.01-1234281020-Summary sheet", page=2)],
            format=[
                self.detected("SCOPE OF THE REVIEW", 1, 100.0),
                self.detected("SUMMARY SHEET", 2, 100.0),
            ],
        )

        rendered = self.render(prompt, headings)

        metadata = '    - (outline metadata) "20.01-1234281020-Summary sheet"'
        assert f"Page 2\n{metadata}\n" in rendered
        assert ']  "SUMMARY SHEET"' in rendered

    def test__a_bookmark_alone_on_its_page_still_earns_a_block(
        self, prompt: SectionReconcilerPrompt
    ) -> None:
        """A page the typography missed is still worth showing when a bookmark points at it."""
        headings = HeadingCandidates(
            outline=[OutlineCandidate(text="Summary sheet7873", page=29)],
            format=[self.detected("SCOPE OF THE REVIEW", 1, 100.0)],
        )

        rendered = self.render(prompt, headings)

        assert 'Page 29\n    - (outline metadata) "Summary sheet7873"' in rendered

    def test__a_wordless_bookmark_is_dropped(self, prompt: SectionReconcilerPrompt) -> None:
        """An enumerator or a bare file code names nothing the page does not already show."""
        headings = HeadingCandidates(
            outline=[
                OutlineCandidate(text=text, page=1)
                for text in ["4", "12.11", "TR-260420-1.1", "Appendix A"]
            ],
            format=[self.detected("APPENDIX A", 1, 100.0)],
        )

        rendered = self.render(prompt, headings)

        assert rendered.count("(outline metadata)") == 1
        assert '"Appendix A"' in rendered

    def test__a_long_bookmark_title_is_clipped_at_the_injected_limit(
        self, reconciler_config: SectionReconcilerConfig
    ) -> None:
        """The limit caps the metadata's bulk, and the clip is visible as a trailing ellipsis."""
        title = "Slide 51 Criteria for assessing a request to decommission (extract)"
        clipped = SectionReconcilerPrompt(
            reconciler_config.model_copy(update={"outline_title_limit": 20})
        )
        headings = HeadingCandidates(
            outline=[OutlineCandidate(text=title, page=1)],
            format=[self.detected("DECOMMISSIONING", 1, 100.0)],
        )

        rendered = self.render(clipped, headings)

        assert '- (outline metadata) "Slide 51 Criteria fo…"' in rendered
        assert "decommission (extract)" not in rendered

    def test__a_whole_document_opens_straight_on_its_first_page(
        self, prompt: SectionReconcilerPrompt
    ) -> None:
        """A document short enough for one window renders exactly as it did before the cut."""
        headings = HeadingCandidates(outline=[], format=[self.detected("FINANCES", 1, 79.2)])

        rendered = self.render(prompt, headings)

        assert rendered.startswith("Page 1\n")
        assert "Window" not in rendered

    def test__a_window_of_a_long_document_names_the_pages_it_answers_for(
        self, prompt: SectionReconcilerPrompt
    ) -> None:
        """One sentence opens the message, and only the window's own pages follow it."""
        headings = HeadingCandidates(
            outline=[],
            format=[self.detected(f"Heading {page}", page, 79.2) for page in [200, 201, 401]],
        )
        middle = PageWindow.tile(headings, size=200)[1]

        rendered = prompt.user(middle, None)

        assert rendered.startswith(
            "Window 2 of 3 of one document: report the sections opening on pages 201 to 400 "
            "only.\n\nPage 201\n"
        )
        assert "Page 200" not in rendered
        assert "Page 401" not in rendered
