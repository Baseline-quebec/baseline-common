"""Tests of the relevance fallback's prompt: the project's criterion, the generic rules."""

from baseline_core.core.llm.document_extraction.classification.section_relevance.prompt import (
    SectionRelevancePrompt,
)
from baseline_core.core.llm.document_extraction.section import LocatedSection, SectionRelevance


class TestSectionRelevancePrompt:
    def test__the_projects_criterion_is_injected_into_the_rules(self, criterion: str) -> None:
        """The model judges against what the project seeks, inside its own tag."""
        system = SectionRelevancePrompt.system(f"  {criterion}\n")

        assert f"<relevance_criterion>\n    {criterion}\n</relevance_criterion>" in system
        assert "{relevance_criterion}" not in system

    def test__a_criterion_with_braces_is_taken_verbatim(self) -> None:
        """The criterion is data: a brace in it is never read as a placeholder."""
        system = SectionRelevancePrompt.system("A section citing a {code}.")

        assert "A section citing a {code}." in system

    def test__doubt_never_settles_on_relevant(self, criterion: str) -> None:
        """The asymmetry holds whatever the criterion: in doubt, uncertain."""
        assert "In doubt, uncertain — NEVER relevant." in SectionRelevancePrompt.system(criterion)

    def test__the_message_quotes_the_heading_then_the_window(self) -> None:
        """The heading as printed, then the window, verbatim."""
        section = LocatedSection(
            title="PUMP REPLACEMENT", start_page=1, level=1, relevance=SectionRelevance.UNCERTAIN
        )

        message = SectionRelevancePrompt.user(section, "The pump of line 2.")

        assert message == (
            'Section heading: "PUMP REPLACEMENT"\n\nText printed under it:\nThe pump of line 2.'
        )
