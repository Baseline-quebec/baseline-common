"""Tests of the extraction service: text always, held to the harvest, vision where pixels speak."""

import asyncio
from typing import Any, ClassVar

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from baseline_core.core.llm.document_extraction.extraction.extraction_service import (
    ExtractionConfig,
    ExtractionService,
)
from baseline_core.core.llm.document_extraction.extraction.mention_harvester import (
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
from baseline_core.core.llm.document_extraction.layout.layout import BBox
from baseline_core.core.llm.document_extraction.layout.layout_parser import LayoutParser
from baseline_core.core.llm.document_extraction.section import (
    LocatedSection,
    ResolvedSection,
    SectionRelevance,
)
from baseline_core.core.llm.document_extraction.sheet_parser import SheetParser
from tests.core.llm.document_extraction.extraction.report_entities import ReportEntities
from tests.core.llm.document_extraction.extraction.scripted_model import ScriptedModel


class Extraction:
    """Builds the service around scripted agents and runs it over one real file."""

    @staticmethod
    def reader(args: dict[str, Any], seen: list[str] | None = None) -> SectionEntitiesAgent[Any]:
        """A text agent whose model answers ``args``, recording its prompts in ``seen``."""
        return SectionEntitiesAgent(
            ScriptedModel.answering(args, seen),
            ReportEntities,
            SectionEntitiesConfig(kinds=ReportEntities.KINDS),
        )

    @staticmethod
    def seer(args: dict[str, Any], calls: list[int]) -> SectionVisionAgent[Any]:
        """A vision agent whose model answers ``args``, counting its calls in ``calls``."""

        def answer(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)]
            )

        return SectionVisionAgent(
            FunctionModel(answer), ReportEntities, SectionVisionConfig(kinds=ReportEntities.KINDS)
        )

    @staticmethod
    def run(
        service: ExtractionService[Any], sections: list[LocatedSection], content: bytes
    ) -> list[ResolvedSection]:
        """Parse the file as the pipeline does (a workbook or a PDF), then extract."""
        document = (
            SheetParser.parse(content)
            if SheetParser.recognizes(content)
            else LayoutParser().parse(content)
        )
        return asyncio.run(service.extract(sections, document, content))


class TestExtract:
    """A text page: read by the text agent alone, its answer held to what it prints."""

    TEXT: ClassVar[str] = "Background\nOrder REF-2025-777 covers the pump for the north site."
    SECTIONS: ClassVar[list[LocatedSection]] = [
        LocatedSection(
            title="Background",
            start_page=1,
            level=1,
            relevance=SectionRelevance.RELEVANT,
            bbox=BBox(x0=72.0, y0=66.5, x1=200.0, y1=80.0),
            text=TEXT,
            matched=True,
        ),
        LocatedSection(
            title="Methodology",
            start_page=1,
            level=1,
            relevance=SectionRelevance.IRRELEVANT,
            bbox=BBox(x0=72.0, y0=123.5, x1=200.0, y1=137.0),
            text="Methodology\nOrdinary body text continues along the page for a while here.",
        ),
    ]
    ANSWER: ClassVar[dict[str, Any]] = {
        "mentions": [
            {"kind": "reference", "value": "REF-2025-777", "role": "own"},
            {"kind": "site", "value": "north site", "role": "own"},
        ],
        "summary": "A pump order.",
    }

    def test__a_section_carries_what_it_names_beside_where_it_was_found(
        self,
        harvester: MentionHarvester,
        unseen: SectionVisionAgent[ReportEntities],
        no_outline_bytes: bytes,
    ) -> None:
        """One record: the located section, and the project's reading of it."""
        service = ExtractionService(
            Extraction.reader(self.ANSWER), unseen, harvester, ExtractionConfig()
        )

        resolved = Extraction.run(service, self.SECTIONS, no_outline_bytes)[0]

        assert (resolved.title, resolved.matched, resolved.bbox) == (
            "Background",
            True,
            self.SECTIONS[0].bbox,
        )
        assert isinstance(resolved.entities, ReportEntities)
        assert [m.value for m in resolved.entities.mentions] == ["REF-2025-777", "north site"]
        assert resolved.entities.summary == "A pump order."
        assert resolved.extraction_error is None

    def test__the_harvest_reaches_the_prompt_and_bounds_the_answer(
        self,
        harvester: MentionHarvester,
        unseen: SectionVisionAgent[ReportEntities],
        no_outline_bytes: bytes,
    ) -> None:
        """The prompt lists the harvested code; a mistyped one is retyped, an invented dropped."""
        seen: list[str] = []
        reader = Extraction.reader(
            {
                "mentions": [
                    {"kind": "site", "value": "REF-2025-777", "role": "own"},
                    {"kind": "reference", "value": "REF-2024-999", "role": "cited"},
                ]
            },
            seen,
        )
        service = ExtractionService(reader, unseen, harvester, ExtractionConfig())

        [resolved, _] = Extraction.run(service, self.SECTIONS, no_outline_bytes)

        assert "<candidate_codes>\nREF-2025-777\t(reference)\n</candidate_codes>" in seen[0]
        assert [(m.kind, m.value) for m in resolved.entities.mentions] == [
            ("reference", "REF-2025-777")
        ]

    def test__a_value_the_section_does_not_print_is_dropped(
        self,
        harvester: MentionHarvester,
        unseen: SectionVisionAgent[ReportEntities],
        no_outline_bytes: bytes,
    ) -> None:
        """A kind without formats is not bounded by the harvest, only by the printed text."""
        reader = Extraction.reader({
            "mentions": [
                {"kind": "site", "value": "north site", "role": "own"},
                {"kind": "supplier", "value": "Acme Tools", "role": "own"},
            ]
        })
        service = ExtractionService(reader, unseen, harvester, ExtractionConfig())

        resolved = Extraction.run(service, self.SECTIONS, no_outline_bytes)[0]

        assert [m.value for m in resolved.entities.mentions] == ["north site"]

    def test__an_irrelevant_section_is_never_handed_to_any_agent(
        self,
        harvester: MentionHarvester,
        unread: SectionEntitiesAgent[ReportEntities],
        unseen: SectionVisionAgent[ReportEntities],
        no_outline_bytes: bytes,
    ) -> None:
        """Irrelevant is terminal: nothing downstream reads it, so no call is spent on it."""
        service = ExtractionService(unread, unseen, harvester, ExtractionConfig())

        [resolved] = Extraction.run(service, [self.SECTIONS[1]], no_outline_bytes)

        assert resolved.relevance is SectionRelevance.IRRELEVANT
        assert resolved.entities.mentions == []

    def test__a_section_placed_nowhere_costs_no_call(
        self,
        harvester: MentionHarvester,
        unread: SectionEntitiesAgent[ReportEntities],
        unseen: SectionVisionAgent[ReportEntities],
        no_outline_bytes: bytes,
    ) -> None:
        """No position, no band: an unplaced section is kept but never read."""
        unplaced = self.SECTIONS[0].model_copy(update={"bbox": None})
        service = ExtractionService(unread, unseen, harvester, ExtractionConfig())

        assert [s.title for s in Extraction.run(service, [unplaced], no_outline_bytes)] == [
            "Background"
        ]

    def test__no_section_no_call(
        self,
        harvester: MentionHarvester,
        unread: SectionEntitiesAgent[ReportEntities],
        unseen: SectionVisionAgent[ReportEntities],
        no_outline_bytes: bytes,
    ) -> None:
        """A document whose plan is empty costs nothing."""
        service = ExtractionService(unread, unseen, harvester, ExtractionConfig())

        assert Extraction.run(service, [], no_outline_bytes) == []

    def test__the_sections_come_back_in_their_order(
        self,
        harvester: MentionHarvester,
        unseen: SectionVisionAgent[ReportEntities],
        no_outline_bytes: bytes,
    ) -> None:
        """Reading concurrently must not reorder the sections under their readings."""
        service = ExtractionService(
            Extraction.reader(self.ANSWER), unseen, harvester, ExtractionConfig()
        )

        resolved = Extraction.run(service, self.SECTIONS, no_outline_bytes)

        assert [s.title for s in resolved] == ["Background", "Methodology"]


class TestUnreadableSection:
    def test__one_unreadable_section_never_fails_its_document(
        self,
        harvester: MentionHarvester,
        unseen: SectionVisionAgent[ReportEntities],
        no_outline_bytes: bytes,
    ) -> None:
        """A reply that never closed its JSON keeps its section, empty, saying why."""

        def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            message = "Exceeded maximum output retries (1)"
            raise UnexpectedModelBehavior(message)

        reader = SectionEntitiesAgent(
            FunctionModel(fail), ReportEntities, SectionEntitiesConfig(kinds=ReportEntities.KINDS)
        )
        service = ExtractionService(reader, unseen, harvester, ExtractionConfig())

        resolved = Extraction.run(service, TestExtract.SECTIONS, no_outline_bytes)

        assert [s.title for s in resolved] == ["Background", "Methodology"]
        assert resolved[0].entities.mentions == []
        assert resolved[0].extraction_error is not None
        assert resolved[0].extraction_error.startswith("UnexpectedModelBehavior")
        assert resolved[1].extraction_error is None

    def test__a_failed_vision_call_is_recorded_beside_the_text_reading(
        self, harvester: MentionHarvester, scanned_page_pdf: bytes
    ) -> None:
        """The text reading stands; the vision call that failed says why in the section."""

        def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            message = "Exceeded maximum output retries (1)"
            raise UnexpectedModelBehavior(message)

        seer = SectionVisionAgent(
            FunctionModel(fail), ReportEntities, SectionVisionConfig(kinds=ReportEntities.KINDS)
        )
        reader = Extraction.reader({
            "mentions": [{"kind": "reference", "value": "REF-2025-777", "role": "own"}]
        })
        service = ExtractionService(reader, seer, harvester, ExtractionConfig())
        text = "Site photographs\nOrder REF-2025-777, photographed on delivery."

        resolved = Extraction.run(service, TestVision.scan_section(text), scanned_page_pdf)[1]

        assert [m.value for m in resolved.entities.mentions] == ["REF-2025-777"]
        assert resolved.extraction_error is not None
        assert resolved.extraction_error.startswith("UnexpectedModelBehavior")


class TestVision:
    """Only an image-heavy page is rendered for the eye, and the harvest bounds what it reads."""

    @staticmethod
    def scan_section(text: str) -> list[LocatedSection]:
        """A text section on page 1, then a section opening on the scanned page 2."""
        return [
            LocatedSection(
                title="Summary",
                start_page=1,
                level=1,
                relevance=SectionRelevance.RELEVANT,
                bbox=BBox(x0=72.0, y0=61.6, x1=200.0, y1=75.0),
                text="Summary",
            ),
            LocatedSection(
                title="Site photographs",
                start_page=2,
                level=1,
                relevance=SectionRelevance.RELEVANT,
                bbox=BBox(x0=72.0, y0=0.0, x1=200.0, y1=10.0),
                text=text,
            ),
        ]

    def test__a_scan_is_read_by_the_eye_and_its_values_stand(
        self, harvester: MentionHarvester, scanned_page_pdf: bytes
    ) -> None:
        """No text layer bounds a scan: what the vision agent reads is merged as read."""
        calls: list[int] = []
        seer = Extraction.seer(
            {"mentions": [{"kind": "reference", "value": "REF-2024-001", "role": "own"}]}, calls
        )
        service = ExtractionService(Extraction.reader({}), seer, harvester, ExtractionConfig())

        first, second = Extraction.run(service, self.scan_section(""), scanned_page_pdf)

        assert len(calls) == 1
        assert first.entities.mentions == []
        assert [m.value for m in second.entities.mentions] == ["REF-2024-001"]

    def test__on_a_section_with_text_the_harvest_bounds_the_vision_codes(
        self, harvester: MentionHarvester, scanned_page_pdf: bytes
    ) -> None:
        """The eye's unharvested code is dropped; its supplier, a kind no format knows, stands."""
        calls: list[int] = []
        seer = Extraction.seer(
            {
                "mentions": [
                    {"kind": "reference", "value": "REF-2024-001", "role": "own"},
                    {"kind": "supplier", "value": "Acme Tools", "role": "own"},
                ]
            },
            calls,
        )
        reader = Extraction.reader({
            "mentions": [{"kind": "reference", "value": "REF-2025-777", "role": "own"}]
        })
        service = ExtractionService(reader, seer, harvester, ExtractionConfig())
        text = "Site photographs\nOrder REF-2025-777, photographed on delivery. " * 5

        resolved = Extraction.run(service, self.scan_section(text), scanned_page_pdf)[1]

        assert [m.value for m in resolved.entities.mentions] == ["REF-2025-777", "Acme Tools"]

    def test__a_text_page_never_reaches_the_eye(
        self,
        harvester: MentionHarvester,
        unseen: SectionVisionAgent[ReportEntities],
        no_outline_bytes: bytes,
    ) -> None:
        """The fixture page prints text and paints nothing: the vision agent is never called."""
        service = ExtractionService(
            Extraction.reader(TestExtract.ANSWER), unseen, harvester, ExtractionConfig()
        )

        resolved = Extraction.run(service, TestExtract.SECTIONS, no_outline_bytes)[0]

        assert len(resolved.entities.mentions) == 2


class TestWorkbook:
    """A workbook paints no pixels: its sections are read off the text of their cells."""

    SHEET: ClassVar[LocatedSection] = LocatedSection(
        title="Statement",
        start_page=1,
        level=1,
        relevance=SectionRelevance.RELEVANT,
        text="Statement\nCash and investments 100 200, order REF-2025-777",
        matched=True,
    )

    def test__a_sheet_section_is_read_as_text_and_held_to_its_harvest(
        self,
        harvester: MentionHarvester,
        unseen: SectionVisionAgent[ReportEntities],
        workbook_bytes: bytes,
    ) -> None:
        """The same layers as a page, without the eye: an invented code does not survive."""
        reader = Extraction.reader({
            "mentions": [
                {"kind": "reference", "value": "REF-2025-777", "role": "own"},
                {"kind": "reference", "value": "REF-2031-000", "role": "cited"},
            ]
        })
        service = ExtractionService(reader, unseen, harvester, ExtractionConfig())

        [resolved] = Extraction.run(service, [self.SHEET], workbook_bytes)

        assert [m.value for m in resolved.entities.mentions] == ["REF-2025-777"]

    def test__an_irrelevant_or_bare_sheet_section_costs_no_call(
        self,
        harvester: MentionHarvester,
        unread: SectionEntitiesAgent[ReportEntities],
        unseen: SectionVisionAgent[ReportEntities],
        workbook_bytes: bytes,
    ) -> None:
        """Irrelevant is terminal on a sheet as on a page; a heading alone names nothing."""
        skipped = self.SHEET.model_copy(update={"relevance": SectionRelevance.IRRELEVANT})
        bare = self.SHEET.model_copy(update={"text": "Statement"})
        service = ExtractionService(unread, unseen, harvester, ExtractionConfig())

        resolved = Extraction.run(service, [skipped, bare], workbook_bytes)

        assert [s.entities.mentions for s in resolved] == [[], []]


def test__sections_are_read_under_the_concurrency_bound(
    harvester: MentionHarvester,
    unseen: SectionVisionAgent[ReportEntities],
    no_outline_bytes: bytes,
) -> None:
    """At most ``max_concurrent_sections`` text calls are in flight at once."""
    in_flight, peak = 0, 0

    async def answer(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args={})])

    reader = SectionEntitiesAgent(
        FunctionModel(answer), ReportEntities, SectionEntitiesConfig(kinds=ReportEntities.KINDS)
    )
    sections = [
        TestExtract.SECTIONS[0].model_copy(
            update={"title": f"S{i}", "bbox": BBox(x0=72.0, y0=60.0 + 20 * i, x1=200.0, y1=70.0)}
        )
        for i in range(6)
    ]
    service = ExtractionService(
        reader, unseen, harvester, ExtractionConfig(max_concurrent_sections=2)
    )

    Extraction.run(service, sections, no_outline_bytes)

    assert peak == 2
