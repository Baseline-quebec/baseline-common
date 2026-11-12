"""SerialItemSplitter: a serial item a section swallowed is split out, a cited one is not."""

from typing import ClassVar

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    FormatCandidate,
    SerialItem,
    Typography,
)
from baseline_core.core.llm.document_extraction.indexing.serial_item_splitter import (
    SerialItemSplitter,
)
from baseline_core.core.llm.document_extraction.layout.layout import BBox
from baseline_core.core.llm.document_extraction.section import LocatedSection, SectionRelevance


class TestSplit:
    ITEMS: ClassVar[list[SerialItem]] = [
        SerialItem(
            code=f"RPT 26-09.{i:02d}",
            candidate=FormatCandidate(
                text=f"RPT 26-09.{i:02d}",
                typography=Typography(size=10.0),
                page=9 + i,
                bbox=BBox(x0=72.0, y0=100.0, x1=400.0, y1=112.0),
                position=100.0 / 792.0,
            ),
        )
        for i in range(9, 13)
    ]

    @staticmethod
    def located(title: str, page: int, text: str) -> LocatedSection:
        """Build a located section.

        Args:
            title: Its heading.
            page: Its start page.
            text: The text printed under it.

        Returns:
            The section.
        """
        return LocatedSection(
            title=title, start_page=page, level=2, relevance=SectionRelevance.RELEVANT, text=text
        )

    def test__items_swallowed_by_a_section_are_split_out_in_order(self) -> None:
        """« RPT 26-09.09 » holding 26-09.10 and .11 becomes three sections, in that order."""
        blob = self.located(
            "RPT 26-09.09",
            18,
            "RPT 26-09.09\n… APPROVED\nRPT 26-09.10\n… DEFERRED\nRPT 26-09.11\n… APPROVED",
        )

        sections = SerialItemSplitter.split([blob], self.ITEMS)

        assert [s.title for s in sections] == ["RPT 26-09.09", "RPT 26-09.10", "RPT 26-09.11"]
        assert [s.relevance for s in sections[1:]] == [SectionRelevance.UNCERTAIN] * 2

    def test__a_code_cited_mid_sentence_splits_nothing(self) -> None:
        """« as recorded under item RPT 26-09.10 » cites the item; it does not open it."""
        cited = self.located(
            "3. CROSS-CHECKS",
            2,
            "3. CROSS-CHECKS\nThe inspector cross-checked the readings of item RPT 26-09.10.",
        )

        assert len(SerialItemSplitter.split([cited], self.ITEMS)) == 1

    def test__a_closing_reference_is_never_an_item(self) -> None:
        """A closing « 40.04 1247905019 » ends the item above it; it opens none."""
        closing = SerialItem(
            code="40.04",
            candidate=self.ITEMS[0].candidate.model_copy(
                update={"text": "40.04 1247905019", "page": 4}
            ),
        )
        decision = self.located("Decide", 4, "Decide …\nAPPROVED\n40.04 1247905019\n______")

        assert len(SerialItemSplitter.split([decision], [closing])) == 1
