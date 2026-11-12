"""Unit tests for IndexingService: the find → reconcile → locate flow."""

import asyncio
from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from baseline_core.core.llm.document_extraction.indexing.detection.heading_finder import (
    HeadingFinder,
)
from baseline_core.core.llm.document_extraction.indexing.detection.heading_locator import (
    HeadingLocator,
)
from baseline_core.core.llm.document_extraction.indexing.indexing_service import IndexingService
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.agent import (
    SectionReconcilerAgent,
)
from baseline_core.core.llm.document_extraction.layout.layout import LayoutDocument
from baseline_core.core.llm.document_extraction.sheet_parser import SheetParser


class TestIndex:
    KEPT_SECTION: ClassVar[list[dict[str, Any]]] = [
        {"title": "Kept", "start_page": 1, "level": 1, "relevance": "irrelevant"}
    ]
    # Returned out of page order, to prove the flow preserves the agent's order verbatim.
    UNSORTED_SECTIONS: ClassVar[list[dict[str, Any]]] = [
        {"title": "Third", "start_page": 4, "level": 1, "relevance": "irrelevant"},
        {"title": "First", "start_page": 1, "level": 1, "relevance": "irrelevant"},
        {"title": "Second", "start_page": 2, "level": 2, "relevance": "relevant"},
    ]
    PRINTED_SECTION: ClassVar[list[dict[str, Any]]] = [
        {"title": "Introduction", "start_page": 2, "level": 1, "relevance": "uncertain"}
    ]
    SHEET_SECTION: ClassVar[list[dict[str, Any]]] = [
        {"title": "Actifs financiers", "start_page": 1, "level": 2, "relevance": "relevant"}
    ]

    @pytest.fixture
    def service(
        self,
        make_reconciler: Callable[[list[dict[str, Any]]], SectionReconcilerAgent],
        finder: HeadingFinder,
        heading_locator: HeadingLocator,
    ) -> Callable[[list[dict[str, Any]]], IndexingService]:
        """Return a factory building one service whose reconciler emits the given sections."""

        def build(sections: list[dict[str, Any]]) -> IndexingService:
            return IndexingService(
                reconciler=make_reconciler(sections), finder=finder, locator=heading_locator
            )

        return build

    def test__runs_find_then_reconcile_on_an_outline_bearing_document(
        self,
        service: Callable[[list[dict[str, Any]]], IndexingService],
        with_outline_layout: LayoutDocument,
    ) -> None:
        """Carry the flow end to end when the document also exposes an embedded outline."""
        indexing = service(self.KEPT_SECTION)

        located = asyncio.run(indexing.index(with_outline_layout))

        assert [s.title for s in located] == ["Kept"]

    def test__hands_back_the_agents_sections_in_order(
        self,
        service: Callable[[list[dict[str, Any]]], IndexingService],
        no_outline_layout: LayoutDocument,
    ) -> None:
        """Hand back the agent's sections untouched, preserving its order."""
        indexing = service(self.UNSORTED_SECTIONS)

        located = asyncio.run(indexing.index(no_outline_layout))

        assert [(s.title, s.start_page, s.level) for s in located] == [
            ("Third", 4, 1),
            ("First", 1, 1),
            ("Second", 2, 2),
        ]

    def test__gives_a_reconciled_section_the_text_printed_under_its_heading(
        self,
        service: Callable[[list[dict[str, Any]]], IndexingService],
        with_outline_layout: LayoutDocument,
    ) -> None:
        """A section whose heading the document prints comes back located and with its text."""
        indexing = service(self.PRINTED_SECTION)

        located = asyncio.run(indexing.index(with_outline_layout))[0]

        assert located.bbox is not None
        assert located.text.startswith("Introduction")

    def test__a_workbook_is_located_on_its_rows(
        self, service: Callable[[list[dict[str, Any]]], IndexingService], workbook_bytes: bytes
    ) -> None:
        """A sheet's section is anchored on its heading row and runs to the next one."""
        indexing = service(self.SHEET_SECTION)

        located = asyncio.run(indexing.index(SheetParser.parse(workbook_bytes)))[0]

        assert located.matched is True
        assert located.text.splitlines()[0] == "Actifs financiers"
