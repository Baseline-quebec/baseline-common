"""IndexingService — the plan of a document: detect the headings, reconcile them, locate them."""

from loguru import logger

from baseline_core.core.llm.document_extraction.indexing.detection.heading_finder import (
    HeadingFinder,
)
from baseline_core.core.llm.document_extraction.indexing.detection.heading_locator import (
    HeadingLocator,
)
from baseline_core.core.llm.document_extraction.indexing.detection.sheet_heading_finder import (
    SheetHeadingFinder,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.agent import (
    SectionReconcilerAgent,
)
from baseline_core.core.llm.document_extraction.indexing.serial_item_splitter import (
    SerialItemSplitter,
)
from baseline_core.core.llm.document_extraction.layout.layout import LayoutDocument
from baseline_core.core.llm.document_extraction.section import LocatedSection


class IndexingService:
    """Turns a parsed document into the sections it prints, each placed back on its page."""

    def __init__(
        self, reconciler: SectionReconcilerAgent, finder: HeadingFinder, locator: HeadingLocator
    ) -> None:
        """Bind the reconciliation agent, the page heading finder and the heading locator.

        Args:
            reconciler: Agent that reconciles the candidates into sections.
            finder: Finder reading both candidate sources of a page document.
            locator: Locator placing each reconciled section back on its page.
        """
        self._reconciler = reconciler
        self._finder = finder
        self._locator = locator

    async def index(self, document: LayoutDocument) -> list[LocatedSection]:
        """Index one parsed document into the sections it prints.

        Args:
            document: The parsed document, from any parser producing a layout tree.

        Returns:
            The reconciled sections in the agent's order, each placed and carrying its text.
        """
        if document.origin == "cells":
            sections = await self._reconciler.reconcile(SheetHeadingFinder.scan(document))
            return self._locator.locate(sections, document)

        candidates = self._finder.scan(document)
        sections = await self._reconciler.reconcile(candidates)
        located = self._locator.locate(sections, document)
        items = self._finder.serial_items(candidates.format)
        split = SerialItemSplitter.split(located, items)

        if len(split) > len(located):
            logger.info(
                "indexing: {} serial item(s) split out of their neighbours",
                len(split) - len(located),
            )
            located = self._locator.locate(split, document)

        return located
