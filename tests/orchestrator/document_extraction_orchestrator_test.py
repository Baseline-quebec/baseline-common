"""Tests of the document pipeline: one parse, the plan indexed, then what it names read."""

import asyncio
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from baseline_core.core.llm.document_extraction.classification.classification_service import (
    ClassificationService,
)
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
from baseline_core.core.llm.document_extraction.document_parser import DocumentParser
from baseline_core.core.llm.document_extraction.extraction.extraction_service import (
    ExtractionConfig,
    ExtractionService,
)
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
from baseline_core.core.llm.document_extraction.indexing.detection.heading_finder import (
    HeadingFinder,
    HeadingFinderConfig,
)
from baseline_core.core.llm.document_extraction.indexing.detection.heading_locator import (
    HeadingLocator,
    HeadingLocatorConfig,
)
from baseline_core.core.llm.document_extraction.indexing.indexing_service import IndexingService
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.agent import (
    SectionReconcilerAgent,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.config import (
    SectionReconcilerConfig,
)
from baseline_core.core.llm.document_extraction.section import SectionRelevance
from baseline_core.orchestrator.document_extraction_orchestrator import (
    DocumentExtractionOrchestrator,
    SourceFile,
)
from tests.core.llm.document_extraction.extraction.report_entities import ReportEntities


class Answering:
    """Function models standing in for a server: each answers one fixed object."""

    @staticmethod
    def model(args: dict[str, Any]) -> FunctionModel:
        """Return a model answering ``args`` through its output tool every time.

        Args:
            args: The object the model answers.

        Returns:
            The model.
        """

        def answer(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)]
            )

        return FunctionModel(answer)


@pytest.fixture
def pipeline() -> DocumentExtractionOrchestrator[ReportEntities]:
    """Return the pipeline over stub models: one uncertain section, read as text."""
    criterion = "the section concerns an intervention on a machine"
    reconciler = SectionReconcilerAgent(
        model=Answering.model({
            "sections": [{"title": "Kept", "start_page": 1, "level": 1, "relevance": "uncertain"}]
        }),
        config=SectionReconcilerConfig(relevance_criterion=criterion),
    )
    indexing = IndexingService(
        reconciler, HeadingFinder(HeadingFinderConfig()), HeadingLocator(HeadingLocatorConfig())
    )
    relevance = SectionRelevanceAgent(
        model=Answering.model({"response": "relevant"}),
        config=SectionRelevanceConfig(relevance_criterion=criterion),
    )
    reader = SectionEntitiesAgent(
        Answering.model({
            "summary": "Essais de pression",
            "mentions": [{"kind": "reference", "value": "BT-2026-0145", "role": "own"}],
        }),
        ReportEntities,
        SectionEntitiesConfig(kinds=ReportEntities.KINDS),
    )
    seer = SectionVisionAgent(
        TestModel(), ReportEntities, SectionVisionConfig(kinds=ReportEntities.KINDS)
    )
    harvester = MentionHarvester([MentionFormat(kind="reference", pattern=r"BT-\d{4}-\d{4}")])
    extraction = ExtractionService(reader, seer, harvester, ExtractionConfig())
    repair_config = SectionRepairConfig(identity_kinds=["reference"])
    classification = ClassificationService(
        relevance,
        SectionRepairAgent(TestModel(), config=repair_config),
        MergeCandidateDetector(repair_config),
    )

    return DocumentExtractionOrchestrator(DocumentParser(), indexing, classification, extraction)


class TestRun:
    def test__an_uncertain_section__is_settled_then_read(
        self, pipeline: DocumentExtractionOrchestrator[ReportEntities], no_outline_bytes: bytes
    ) -> None:
        """The heading left it uncertain; its text settles it, and only then is it read."""
        file = SourceFile(content=no_outline_bytes, name="no_outline.pdf")

        document = asyncio.run(pipeline.run(file))

        (section,) = document.sections
        assert section.relevance is SectionRelevance.RELEVANT
        assert isinstance(section.entities, ReportEntities)
        assert section.entities.summary == "Essais de pression"

    def test__an_invented_code__never_survives_the_harvest(
        self, pipeline: DocumentExtractionOrchestrator[ReportEntities], no_outline_bytes: bytes
    ) -> None:
        """The model answers « BT-2026-0145 »; the file prints no such code, so it is dropped."""
        file = SourceFile(content=no_outline_bytes, name="no_outline.pdf")

        document = asyncio.run(pipeline.run(file))

        assert document.sections[0].entities.mentions == []

    def test__the_files_own_facts_are_read_off_the_parse(
        self, pipeline: DocumentExtractionOrchestrator[ReportEntities], with_outline_bytes: bytes
    ) -> None:
        """The metadata comes from the tree, and the id defaults to the checksum."""
        file = SourceFile(content=with_outline_bytes, name="with_outline.pdf")

        document = asyncio.run(pipeline.run(file))

        assert document.file.page_count == 4
        assert document.file.has_embedded_outline is True
        assert document.document_id == document.file.checksum

    def test__a_workbook_runs_through_on_its_cells(
        self, pipeline: DocumentExtractionOrchestrator[ReportEntities], workbook_bytes: bytes
    ) -> None:
        """A sheet is parsed by the sheet parser, indexed and read as text."""
        file = SourceFile(content=workbook_bytes, name="inventaire.xlsx", document_id="inv")

        document = asyncio.run(pipeline.run(file))

        assert document.document_id == "inv"
        assert document.text.startswith("Situation financière")
        assert [section.title for section in document.sections] == ["Kept"]
