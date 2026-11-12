"""Fixtures of the classification tests: both judges, the detector, a section maker."""

from collections.abc import Callable

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from baseline_core.core.llm.document_extraction.classification.section_relevance.agent import (
    SectionRelevanceAgent,
)
from baseline_core.core.llm.document_extraction.classification.section_relevance.config import (
    SectionRelevanceConfig,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.agent import (
    SectionRepairAgent,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.config import (
    SectionRepairConfig,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.merge_candidate_detector import (
    MergeCandidateDetector,
)
from baseline_core.core.llm.document_extraction.section import ResolvedSection, SectionRelevance
from baseline_core.core.llm.document_extraction.section_entities import Mention, SectionEntities


@pytest.fixture
def config() -> SectionRepairConfig:
    """Return a config where a contract number and a work order identify an item."""
    return SectionRepairConfig(identity_kinds=["contract", "work_order"])


@pytest.fixture
def detector(config: SectionRepairConfig) -> MergeCandidateDetector:
    """Return the detector on the test config."""
    return MergeCandidateDetector(config)


@pytest.fixture
def boundary_agent(config: SectionRepairConfig) -> SectionRepairAgent:
    """Return a boundary agent whose model is never really called."""
    return SectionRepairAgent(TestModel(), config=config)


@pytest.fixture
def make_section() -> Callable[..., ResolvedSection]:
    """Return a factory building one section naming the given mentions over the given text."""

    def build(
        title: str,
        *,
        mentions: list[Mention] | None = None,
        text: str = "",
        relevance: SectionRelevance = SectionRelevance.RELEVANT,
    ) -> ResolvedSection:
        return ResolvedSection(
            title=title,
            start_page=1,
            level=2,
            relevance=relevance,
            text=text,
            entities=SectionEntities(mentions=mentions or []),
        )

    return build


@pytest.fixture
def criterion() -> str:
    """Return a project's relevance criterion, in its own words."""
    return "A repair or a replacement of equipment at an identified site."


@pytest.fixture
def stub_relevance(criterion: str) -> SectionRelevanceAgent:
    """Return a relevance agent whose model settles every uncertain section as relevant."""

    def settle(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name=info.output_tools[0].name, args={"response": "relevant"})
            ]
        )

    return SectionRelevanceAgent(
        model=FunctionModel(settle), config=SectionRelevanceConfig(relevance_criterion=criterion)
    )
