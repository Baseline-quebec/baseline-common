"""Tests of the text agent's system prompt: the project's kinds, the rules, the list's scope."""

from baseline_core.core.llm.document_extraction.extraction.section_entities.prompt import (
    SectionEntitiesPrompt,
)
from tests.core.llm.document_extraction.extraction.report_entities import ReportEntities


class TestSectionEntitiesPrompt:
    """The system prompt is one static prefix per project."""

    def test__the_project_s_kinds_are_listed_with_their_description(self) -> None:
        """The model may give a mention only the kinds the project lists."""
        system = SectionEntitiesPrompt.system(ReportEntities.KINDS)

        assert "    - site: a place where the work happens\n" in system
        assert "    - supplier: an organization that delivers goods or services\n" in system

    def test__the_candidate_list_is_scoped_to_its_own_kinds(self) -> None:
        """Regression: a list read as « report only codes » emptied every other field."""
        system = SectionEntitiesPrompt.system(ReportEntities.KINDS)

        assert "concerns the ``mentions`` of the\n    kinds it shows ALONE" in system
        assert "Every\n    other field" in system

    def test__the_field_descriptions_are_the_project_s_instructions(self) -> None:
        """What a project field means lives in its schema, which the prompt points at."""
        system = SectionEntitiesPrompt.system(ReportEntities.KINDS)

        assert "the project's instruction for that field" in system
