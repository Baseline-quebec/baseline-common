"""VisionRouter — cuts each section out of the pages it runs across, for the eye to read."""

from dataclasses import dataclass

from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    LayoutDocument,
    LayoutPage,
    PageRegion,
)
from baseline_core.core.llm.document_extraction.section import LocatedSection, SectionRelevance


@dataclass(frozen=True)
class SectionSpan:
    """The run of one section: from its heading down to where the next one opens."""

    start_page: int
    top: float
    end_page: int
    bottom: float | None
    """Height the next heading opens at on ``end_page``; None to run to the page's end."""


class VisionRouter:
    """Cuts every section worth reading out of its pages, header and footer left behind."""

    @classmethod
    def route(
        cls, sections: list[LocatedSection], document: LayoutDocument
    ) -> list[list[PageRegion]]:
        """Cut every section the pipeline still reads out of the pages it runs across.

        Args:
            sections: The indexed sections, in reconciliation order.
            document: The parsed tree the sections index into.

        Returns:
            One list of bands per section, in their order; empty for an irrelevant or unplaced one.
        """
        if document.origin == "cells":
            return [[] for _ in sections]

        pages = {page.number: page for page in document.pages}
        placed = sorted(
            (section.start_page, section.bbox.y0, rank)
            for rank, section in enumerate(sections)
            if section.bbox is not None and section.start_page in pages
        )
        spans: list[list[PageRegion]] = [[] for _ in sections]
        last = max(pages, default=0)

        for order, (start_page, top, rank) in enumerate(placed):
            if sections[rank].relevance is SectionRelevance.IRRELEVANT:
                continue

            following = placed[order + 1] if order + 1 < len(placed) else None
            end_page, bottom = (following[0], following[1]) if following else (last, None)

            # A section opening its page owns what prints above it, such as a form's letterhead.
            if (
                following is not None
                and end_page > start_page
                and (sections[following[2]].opener_anchored or cls._sheet(pages[end_page]))
            ):
                end_page, bottom = end_page - 1, None

            span = SectionSpan(start_page=start_page, top=top, end_page=end_page, bottom=bottom)
            spans[rank] = cls._bands(span, pages)

        return spans

    @classmethod
    def _bands(cls, span: SectionSpan, pages: dict[int, LayoutPage]) -> list[PageRegion]:
        """Cut the run between two headings into one band per page it crosses.

        Args:
            span: Where the section opens, and where the next one does.
            pages: The document's pages, by number.

        Returns:
            One band per page crossed, each already clear of the running header and footer.
        """
        bands: list[PageRegion] = []

        for number in range(span.start_page, span.end_page + 1):
            page = pages.get(number)

            if page is None:
                continue

            head, foot = page.content_band
            # A drawing sheet is one drawing: it is never split at a heading.
            sheet = cls._sheet(page)
            y0 = max(span.top, head) if number == span.start_page and not sheet else head
            y1 = (
                min(span.bottom, foot)
                if number == span.end_page and span.bottom is not None and not sheet
                else foot
            )

            if y1 <= y0:
                continue

            bands.append(
                PageRegion(page=number, clip=BBox(x0=0.0, y0=y0, x1=page.size.width, y1=y1))
            )

        return bands

    @staticmethod
    def _sheet(page: LayoutPage) -> bool:
        """Whether the page is a landscape drawing sheet, read whole rather than split.

        Args:
            page: The page a cut would fall on.

        Returns:
            True when the page prints wider than tall — a drawing sheet, its caption naming it.
        """
        return page.size.width > page.size.height
