"""Tests of the text agent: what it renders, and what it hands back."""

import asyncio
from typing import Any, ClassVar

from baseline_core.core.llm.document_extraction.extraction.mention_harvester import Candidate
from baseline_core.core.llm.document_extraction.extraction.section_entities.agent import (
    SectionEntitiesAgent,
)
from baseline_core.core.llm.document_extraction.extraction.section_entities.config import (
    SectionEntitiesConfig,
)
from baseline_core.core.llm.document_extraction.section import LocatedSection, SectionRelevance
from baseline_core.core.llm.document_extraction.section_entities import MentionRole
from tests.core.llm.document_extraction.extraction.report_entities import ReportEntities
from tests.core.llm.document_extraction.extraction.scripted_model import ScriptedModel


class TestSectionEntitiesAgent:
    """The prompt the agent builds, and the reading it returns from a canned answer."""

    SECTION: ClassVar[LocatedSection] = LocatedSection(
        title="Delivery of the pump assembly to the north site",
        start_page=1,
        level=1,
        relevance=SectionRelevance.RELEVANT,
        text="Delivery of the pump assembly to the north site\n"
        "ORDER REF-2026-313 replaces order REF-2026-181.",
    )
    ANSWER: ClassVar[dict[str, Any]] = {
        "mentions": [
            {"kind": "reference", "value": "REF-2026-313", "role": "own"},
            {"kind": "reference", "value": "REF-2026-181", "role": "cited"},
        ],
        "summary": "A pump delivery.",
    }

    def agent(
        self, seen: list[str], body_char_limit: int = 12_000, candidate_limit: int = 60
    ) -> SectionEntitiesAgent[ReportEntities]:
        """Build an agent whose model records the prompt and answers ``ANSWER``."""
        return SectionEntitiesAgent(
            ScriptedModel.answering(self.ANSWER, seen),
            ReportEntities,
            SectionEntitiesConfig(
                kinds=ReportEntities.KINDS,
                body_char_limit=body_char_limit,
                candidate_limit=candidate_limit,
            ),
        )

    def test__the_prompt_carries_the_title_and_the_body_apart(self) -> None:
        """The heading and the text under it reach the model as two named parts."""
        seen: list[str] = []

        asyncio.run(self.agent(seen).extract(self.SECTION, []))

        assert f"<section_title>\n{self.SECTION.title}\n</section_title>" in seen[0]
        assert "<section_body>\nORDER REF-2026-313 replaces" in seen[0]

    def test__the_opener_is_named_to_the_model_only_when_the_heading_prints_one(self) -> None:
        """The code the heading opens on reaches the model as its own part."""
        seen: list[str] = []
        opened = self.SECTION.model_copy(update={"section_number": "ITEM 4.2"})

        asyncio.run(self.agent(seen).extract(opened, []))
        asyncio.run(self.agent(seen).extract(self.SECTION, []))

        assert seen[0].startswith("<opener>ITEM 4.2</opener>\n")
        assert "<opener>" not in seen[1]

    def test__a_body_longer_than_the_limit_is_cut_to_it(self) -> None:
        """The cap guards against a runaway appendix, so it must really bound the prompt."""
        seen: list[str] = []
        long = self.SECTION.model_copy(update={"text": "x" * 500})

        asyncio.run(self.agent(seen, body_char_limit=10).extract(long, []))

        assert "<section_body>\n" + "x" * 10 + "\n</section_body>" in seen[0]

    def test__the_title_is_never_cut(self) -> None:
        """Only the body is capped: a truncated heading would lose the anchoring code."""
        seen: list[str] = []

        asyncio.run(self.agent(seen, body_char_limit=1).extract(self.SECTION, []))

        assert self.SECTION.title in seen[0]

    def test__the_candidates_reach_the_user_prompt_capped(self) -> None:
        """The harvest goes in the user message, its list cut to the configured count."""
        seen: list[str] = []
        candidates = [
            Candidate(value=f"REF-2026-{n:03d}", kind="reference", start=n, end=n + 1)
            for n in range(3)
        ]

        asyncio.run(self.agent(seen, candidate_limit=2).extract(self.SECTION, candidates))

        assert (
            "<candidate_codes>\nREF-2026-000\t(reference)\nREF-2026-001\t(reference)\n"
            in (seen[0])
        )
        assert "REF-2026-002" not in seen[0]

    def test__no_candidate_no_list(self) -> None:
        """A section the formats found nothing in is read without an empty list."""
        seen: list[str] = []

        asyncio.run(self.agent(seen).extract(self.SECTION, []))

        assert "<candidate_codes>" not in seen[0]

    def test__it_returns_the_project_s_schema_as_the_model_emitted_it(self) -> None:
        """The output is the project's object, typed — no reshaping in between."""
        entities = asyncio.run(self.agent([]).extract(self.SECTION, []))

        assert isinstance(entities, ReportEntities)
        assert [(m.value, m.role) for m in entities.mentions] == [
            ("REF-2026-313", MentionRole.OWN),
            ("REF-2026-181", MentionRole.CITED),
        ]
        assert entities.summary == "A pump delivery."

    def test__a_section_naming_nothing_yields_an_empty_object(self) -> None:
        """An empty answer survives the round trip: it is a correct verdict, not a failure."""
        agent = SectionEntitiesAgent(
            ScriptedModel.answering({}),
            ReportEntities,
            SectionEntitiesConfig(kinds=ReportEntities.KINDS),
        )
        bare = self.SECTION.model_copy(update={"title": "Distribution list", "text": ""})

        entities = asyncio.run(agent.extract(bare, []))

        assert entities.mentions == []
        assert entities.summary is None
