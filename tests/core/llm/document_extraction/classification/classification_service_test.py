"""The classification settles uncertain sections and folds split ones back into the one above."""

import asyncio
from collections.abc import Callable
from typing import ClassVar, override

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
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
    Evidence,
    MergeCandidateDetector,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.output import (
    BoundaryDecision,
    BoundaryOpinion,
)
from baseline_core.core.llm.document_extraction.section import (
    FileReference,
    ResolvedSection,
    SectionRelevance,
)
from baseline_core.core.llm.document_extraction.section_entities import Mention, MentionRole
from tests.core.llm.document_extraction.classification.section_repair.report_entities import (
    ReportEntities,
)


class CannedRepairAgent(SectionRepairAgent):
    """A boundary agent answering every boundary the same way, and keeping what it was asked."""

    def __init__(self, decision: BoundaryDecision, config: SectionRepairConfig) -> None:
        """Bind the answer it always gives.

        Args:
            decision: What it returns for every boundary put to it.
            config: The repair's config, as the real agent takes it.
        """
        super().__init__(TestModel(), config=config)
        self._decision = decision
        self.asked: list[list[str]] = []

    @override
    async def judge(
        self, previous: ResolvedSection, current: ResolvedSection, evidence: list[Evidence]
    ) -> BoundaryOpinion:
        """Record the boundary and answer as canned.

        Args:
            previous: The absorbing section, merges already folded in.
            current: The section whose independence is in question.
            evidence: The detector's clues, unread.

        Returns:
            The canned opinion.
        """
        self.asked.append([previous.title, current.title])
        return BoundaryOpinion(decision=self._decision, reason="canned")


class TestSectionRepair:
    """What the fold merges, what it refuses, and what a merged section carries."""

    FIRST: ClassVar[Mention] = Mention(kind="contract", value="C-2026-0313", role=MentionRole.OWN)
    SECOND: ClassVar[Mention] = Mention(kind="contract", value="C-2026-0999", role=MentionRole.OWN)

    @pytest.fixture
    def merging(self, config: SectionRepairConfig) -> CannedRepairAgent:
        """Return an agent merging every boundary the detector brings it."""
        return CannedRepairAgent(BoundaryDecision.MERGE, config)

    @pytest.fixture
    def repair(
        self,
        stub_relevance: SectionRelevanceAgent,
        merging: CannedRepairAgent,
        detector: MergeCandidateDetector,
    ) -> ClassificationService:
        """Return a repair whose agent always merges."""
        return ClassificationService(stub_relevance, merging, detector)

    def test__a_keyed_section_then_two_keyless_absorbs_both(
        self,
        repair: ClassificationService,
        merging: CannedRepairAgent,
        make_section: Callable[..., ResolvedSection],
    ) -> None:
        """« S, W1, W2 »: W2 meets the S+W1 already merged, so the run keeps its host."""
        sections = [
            make_section("REPORT", mentions=[self.FIRST]),
            make_section("CONSIDERING"),
            make_section("RECOMMENDATION"),
        ]

        kept = asyncio.run(repair.repair(sections))

        assert [item.title for item in kept] == ["REPORT"]
        assert merging.asked == [["REPORT", "CONSIDERING"], ["REPORT", "RECOMMENDATION"]]

    def test__a_second_keyed_section_breaks_the_run_and_starts_its_own(
        self,
        repair: ClassificationService,
        merging: CannedRepairAgent,
        make_section: Callable[..., ResolvedSection],
    ) -> None:
        """« S, S1, W, W1 »: two items never merge, and the keyless pair joins the SECOND."""
        sections = [
            make_section("FIRST", mentions=[self.FIRST]),
            make_section("SECOND", mentions=[self.SECOND]),
            make_section("CONSIDERING"),
            make_section("RECOMMENDATION"),
        ]

        kept = asyncio.run(repair.repair(sections))

        assert [item.title for item in kept] == ["FIRST", "SECOND"]
        assert merging.asked == [["SECOND", "CONSIDERING"], ["SECOND", "RECOMMENDATION"]]

    def test__a_boundary_the_agent_refuses_leaves_both_standing(
        self,
        stub_relevance: SectionRelevanceAgent,
        config: SectionRepairConfig,
        detector: MergeCandidateDetector,
        make_section: Callable[..., ResolvedSection],
    ) -> None:
        """The detector only asks; the agent decides, and a refusal ends the run."""
        separating = CannedRepairAgent(BoundaryDecision.SEPARATE, config)
        sections = [
            make_section("REPORT", mentions=[self.FIRST]),
            make_section("CONSIDERING"),
            make_section("RECOMMENDATION"),
        ]

        kept = asyncio.run(
            ClassificationService(stub_relevance, separating, detector).repair(sections)
        )

        # The refused CONSIDERING hosts the last boundary, and it owns no key.
        assert [item.title for item in kept] == ["REPORT", "CONSIDERING", "RECOMMENDATION"]
        assert separating.asked == [["REPORT", "CONSIDERING"]]

    def test__a_failed_judgement_keeps_both_sections(
        self,
        stub_relevance: SectionRelevanceAgent,
        config: SectionRepairConfig,
        detector: MergeCandidateDetector,
        make_section: Callable[..., ResolvedSection],
    ) -> None:
        """Two sections kept apart lose nothing; a refused request must not lose the document."""

        def refuse(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise ModelHTTPError(status_code=503, model_name="test")

        agent = SectionRepairAgent(FunctionModel(refuse), config=config)
        sections = [make_section("REPORT", mentions=[self.FIRST]), make_section("CONSIDERING")]

        kept = asyncio.run(ClassificationService(stub_relevance, agent, detector).repair(sections))

        assert [item.title for item in kept] == ["REPORT", "CONSIDERING"]

    def test__two_sections_each_owning_its_identity_never_reach_the_agent(
        self,
        repair: ClassificationService,
        merging: CannedRepairAgent,
        make_section: Callable[..., ResolvedSection],
    ) -> None:
        """The exclusion holds inside the fold: two items are two items, at no cost."""
        sections = [
            make_section("FIRST", mentions=[self.FIRST]),
            make_section("SECOND", mentions=[self.SECOND]),
        ]

        kept = asyncio.run(repair.repair(sections))

        assert [item.title for item in kept] == ["FIRST", "SECOND"]
        assert merging.asked == []

    def test__a_keyless_section_opening_the_document_has_no_host_to_rejoin(
        self,
        repair: ClassificationService,
        merging: CannedRepairAgent,
        make_section: Callable[..., ResolvedSection],
    ) -> None:
        """``NO_KEY_OF_ITS_OWN`` needs a host, and nothing above the first section is one."""
        sections = [make_section("PREAMBLE"), make_section("OTHER")]

        kept = asyncio.run(repair.repair(sections))

        assert [item.title for item in kept] == ["PREAMBLE", "OTHER"]
        assert merging.asked == []

    def test__an_empty_document_is_repaired_into_nothing(
        self, repair: ClassificationService
    ) -> None:
        """No section is not an error."""
        assert asyncio.run(repair.repair([])) == []


class TestTheFold:
    """What a merged section keeps of each half."""

    FIRST: ClassVar[Mention] = Mention(kind="contract", value="C-2026-0313", role=MentionRole.OWN)
    SITE: ClassVar[Mention] = Mention(kind="site", value="Harbour Road", role=MentionRole.OWN)

    @pytest.fixture
    def repair(
        self,
        stub_relevance: SectionRelevanceAgent,
        config: SectionRepairConfig,
        detector: MergeCandidateDetector,
    ) -> ClassificationService:
        """Return a repair whose agent always merges."""
        return ClassificationService(
            stub_relevance, CannedRepairAgent(BoundaryDecision.MERGE, config), detector
        )

    def test__the_absorbers_place_in_the_plan_survives_the_merge(
        self, repair: ClassificationService, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The absorber is the certain one — it owned a key — so its heading and depth win."""
        absorber = make_section("REPORT", mentions=[self.FIRST])
        trailing = make_section("CONSIDERING").model_copy(update={"level": 3, "start_page": 2})

        [merged] = asyncio.run(repair.repair([absorber, trailing]))

        assert merged.title == "REPORT"
        assert merged.start_page == absorber.start_page
        assert merged.level == absorber.level

    def test__the_absorbed_text_keeps_the_heading_the_page_printed(
        self, repair: ClassificationService, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Nothing printed is dropped: the false heading stays where the page put it."""
        absorber = make_section("REPORT", mentions=[self.FIRST], text="the report")
        trailing = make_section("CONSIDERING", text="CONSIDERING\nthe finding")

        [merged] = asyncio.run(repair.repair([absorber, trailing]))

        assert merged.text == "the report\n\nCONSIDERING\nthe finding"

    def test__what_either_half_named_is_carried_by_the_merge(
        self, repair: ClassificationService, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A mention only the trailing half printed is not lost when the split is undone."""
        absorber = make_section("REPORT", mentions=[self.FIRST])
        trailing = make_section("CONSIDERING", mentions=[self.SITE])

        [merged] = asyncio.run(repair.repair([absorber, trailing]))

        assert merged.entities.mentions == [self.FIRST, self.SITE]

    def test__the_projects_own_fields_merge_with_the_absorber_first(
        self, repair: ClassificationService, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The fold reads the project's schema whole, and the absorber's reading wins."""
        absorber = make_section("REPORT").model_copy(
            update={"entities": ReportEntities(mentions=[self.FIRST])}
        )
        trailing = make_section("CONSIDERING").model_copy(
            update={"entities": ReportEntities(summary="Replaces the pump.")}
        )

        [merged] = asyncio.run(repair.repair([absorber, trailing]))

        assert isinstance(merged.entities, ReportEntities)
        assert merged.entities.summary == "Replaces the pump."

    def test__file_references_are_united_on_their_target(
        self, repair: ClassificationService, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A file both halves link to is listed once, and one only the trailing half links kept."""
        shared = FileReference(target="files/annex-a.pdf", filename="annex-a.pdf")
        own = FileReference(target="files/annex-b.pdf", filename="annex-b.pdf")
        absorber = make_section("REPORT", mentions=[self.FIRST]).model_copy(
            update={"file_references": [shared]}
        )
        trailing = make_section("CONSIDERING").model_copy(
            update={"file_references": [shared, own]}
        )

        [merged] = asyncio.run(repair.repair([absorber, trailing]))

        assert merged.file_references == [shared, own]

    def test__a_half_that_failed_to_read_still_says_so(
        self, repair: ClassificationService, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A merge never hides that part of its text went unread."""
        absorber = make_section("REPORT", mentions=[self.FIRST])
        trailing = make_section("CONSIDERING").model_copy(update={"extraction_error": "timeout"})

        [merged] = asyncio.run(repair.repair([absorber, trailing]))

        assert merged.extraction_error == "timeout"


class TestSettle:
    """An uncertain section is judged on its text; a settled one is left alone."""

    def test__an_uncertain_section_takes_the_judges_answer(
        self,
        stub_relevance: SectionRelevanceAgent,
        boundary_agent: SectionRepairAgent,
        detector: MergeCandidateDetector,
        make_section: Callable[..., ResolvedSection],
    ) -> None:
        """The stub judge settles every uncertain section as relevant."""
        service = ClassificationService(stub_relevance, boundary_agent, detector)
        uncertain = make_section(
            "REPORT",
            text="Pump P-4 replaced at the north site.",
            relevance=SectionRelevance.UNCERTAIN,
        )

        settled = asyncio.run(service.settle([uncertain]))

        assert [section.relevance for section in settled] == [SectionRelevance.RELEVANT]

    def test__a_settled_section_is_never_judged(
        self,
        boundary_agent: SectionRepairAgent,
        detector: MergeCandidateDetector,
        criterion: str,
        make_section: Callable[..., ResolvedSection],
    ) -> None:
        """Only uncertain sections reach the judge; a failing judge proves it is never called."""

        def refuse(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise ModelHTTPError(status_code=503, model_name="test")

        judge = SectionRelevanceAgent(
            model=FunctionModel(refuse),
            config=SectionRelevanceConfig(relevance_criterion=criterion),
        )
        service = ClassificationService(judge, boundary_agent, detector)
        irrelevant = make_section("APPENDIX", relevance=SectionRelevance.IRRELEVANT)

        assert asyncio.run(service.settle([irrelevant])) == [irrelevant]

    def test__a_failed_judgement_leaves_the_section_uncertain(
        self,
        boundary_agent: SectionRepairAgent,
        detector: MergeCandidateDetector,
        criterion: str,
        make_section: Callable[..., ResolvedSection],
    ) -> None:
        """A refused request costs the judgement, never the section."""

        def refuse(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise ModelHTTPError(status_code=503, model_name="test")

        judge = SectionRelevanceAgent(
            model=FunctionModel(refuse),
            config=SectionRelevanceConfig(relevance_criterion=criterion),
        )
        service = ClassificationService(judge, boundary_agent, detector)
        uncertain = make_section(
            "REPORT",
            text="Pump P-4 replaced at the north site.",
            relevance=SectionRelevance.UNCERTAIN,
        )

        assert asyncio.run(service.settle([uncertain])) == [uncertain]
