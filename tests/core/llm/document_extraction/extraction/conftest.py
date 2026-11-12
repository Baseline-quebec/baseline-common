"""Fixtures of the extraction tests: the project's formats, and agents that must not be called."""

import pytest

from baseline_core.core.llm.document_extraction.extraction.mention_harvester import (
    MentionFormat,
    MentionHarvester,
)
from baseline_core.core.llm.document_extraction.extraction.section_entities.agent import (
    SectionEntitiesAgent,
)
from baseline_core.core.llm.document_extraction.extraction.section_entities.config import (
    SectionEntitiesConfig,
)
from baseline_core.core.llm.document_extraction.extraction.section_vision.agent import (
    SectionVisionAgent,
)
from baseline_core.core.llm.document_extraction.extraction.section_vision.config import (
    SectionVisionConfig,
)
from tests.core.llm.document_extraction.extraction.report_entities import ReportEntities
from tests.core.llm.document_extraction.extraction.scripted_model import ScriptedModel


@pytest.fixture
def harvester() -> MentionHarvester:
    """Return a harvester knowing one shape: a report reference « REF-2025-777 »."""
    return MentionHarvester([MentionFormat(kind="reference", pattern=r"REF-\d{4}-\d{3}")])


@pytest.fixture
def entities_config() -> SectionEntitiesConfig:
    """Return the text agent's config for the test project's kinds."""
    return SectionEntitiesConfig(kinds=ReportEntities.KINDS)


@pytest.fixture
def vision_config() -> SectionVisionConfig:
    """Return the vision agent's config for the test project's kinds."""
    return SectionVisionConfig(kinds=ReportEntities.KINDS)


@pytest.fixture
def unread(entities_config: SectionEntitiesConfig) -> SectionEntitiesAgent[ReportEntities]:
    """Return a text agent that must never be reached."""
    return SectionEntitiesAgent(ScriptedModel.silent(), ReportEntities, entities_config)


@pytest.fixture
def unseen(vision_config: SectionVisionConfig) -> SectionVisionAgent[ReportEntities]:
    """Return a vision agent that must never be reached."""
    return SectionVisionAgent(ScriptedModel.silent(), ReportEntities, vision_config)
