"""Tests of the document records: the file's facts, and the date each section resolves to."""

from datetime import date

from baseline_core.core.llm.document_extraction.indexed_document import (
    FileFormat,
    FileMetadata,
    IndexedDocument,
)
from baseline_core.core.llm.document_extraction.layout.layout import LayoutDocument
from baseline_core.core.llm.document_extraction.layout.layout_parser import LayoutParser
from baseline_core.core.llm.document_extraction.section import ResolvedSection, SectionRelevance
from baseline_core.core.llm.document_extraction.section_entities import SectionEntities


class TestFileMetadata:
    def test__from_layout_reads_the_files_facts_off_the_parsed_tree(
        self, no_outline_bytes: bytes
    ) -> None:
        """Fill every derivable field from the tree, without reopening the file."""
        document = LayoutParser().parse(no_outline_bytes)

        metadata = FileMetadata.from_layout(document, "no_outline.pdf", None)

        assert metadata.name == "no_outline.pdf"
        assert metadata.format is FileFormat.PDF
        assert metadata.page_count == document.page_count
        assert metadata.checksum == document.checksum
        assert metadata.has_embedded_outline is False
        assert metadata.text_is_extractable is True

    def test__a_hand_built_tree_carries_no_checksum(self) -> None:
        """Only bytes settle a checksum: a tree built from anything else leaves it empty."""
        tree = LayoutDocument(pages=[], outline=[])

        assert FileMetadata.from_layout(tree, "", None).checksum is None

    def test__a_tree_set_in_cells_is_a_workbook(self) -> None:
        """The sheet parser's tree names its origin, so the file is filed as XLSX, never PDF."""
        sheet = LayoutDocument(origin="cells", pages=[], outline=[])

        assert FileMetadata.from_layout(sheet, "", None).format is FileFormat.XLSX

    def test__the_source_is_kept_as_the_collection_gave_it(self) -> None:
        """A link annotation names the file by this string, so it is stored untouched."""
        tree = LayoutDocument(pages=[], outline=[])
        source = "https://example.org/annexes/SD-2025-2171.xlsx"

        assert FileMetadata.from_layout(tree, "", source).source == source


class TestFileFormat:
    def test__a_workbook_answers_to_both_office_suffixes(self) -> None:
        """A macro-enabled workbook is the same archive; the binary ``.xls`` is not."""
        assert FileFormat.XLSX.suffixes == [".xlsx", ".xlsm"]
        assert ".xls" not in FileFormat.XLSX.suffixes


class TestIndexedDocumentSectionDates:
    @staticmethod
    def section(dated: date | None) -> ResolvedSection:
        """A relevant section stating the date given, or none."""
        return ResolvedSection(
            title="Essais",
            start_page=1,
            level=1,
            relevance=SectionRelevance.RELEVANT,
            entities=SectionEntities(dated=dated),
        )

    def test__a_section_falls_back_on_the_last_date_opened_above_it(self) -> None:
        """Own date first, then the last one above, then the document's own."""
        document = IndexedDocument(
            document_id="report-7",
            file=FileMetadata(name="report-7.pdf"),
            dated=date(2026, 1, 5),
            sections=[self.section(None), self.section(date(2026, 2, 1)), self.section(None)],
        )

        assert document.section_dates == [date(2026, 1, 5), date(2026, 2, 1), date(2026, 2, 1)]
