"""Tests of the relevance fallback's config."""

import pytest
from pydantic import ValidationError

from baseline_core.core.llm.document_extraction.classification.section_relevance.config import (
    SectionRelevanceConfig,
)


class TestSectionRelevanceConfig:
    def test__an_overlap_as_wide_as_the_window_is_refused(self, criterion: str) -> None:
        """A window re-reading all of itself would never move down the text."""
        with pytest.raises(ValidationError, match="overlap_chars"):
            SectionRelevanceConfig(
                relevance_criterion=criterion, window_chars=300, overlap_chars=300
            )

    def test__a_project_must_state_its_criterion(self) -> None:
        """No default criterion: relevance means nothing until the project says what it seeks."""
        with pytest.raises(ValidationError, match="relevance_criterion"):
            SectionRelevanceConfig(relevance_criterion="")
