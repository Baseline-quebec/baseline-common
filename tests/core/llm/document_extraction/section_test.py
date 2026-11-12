"""Tests of the section records: the printed heading, the body, and the file references."""

from typing import ClassVar

from baseline_core.core.llm.document_extraction.layout.layout import BBox, Link
from baseline_core.core.llm.document_extraction.section import (
    FileReference,
    LocatedSection,
    ResolvedSection,
    Section,
    SectionRelevance,
)
from baseline_core.core.llm.document_extraction.section_entities import (
    Mention,
    MentionRole,
    SectionEntities,
)


class TestSectionPrintedHeading:
    """``printed_heading`` restores the line the reconciler split in two."""

    def test__the_opener_is_restored_before_the_title(self) -> None:
        """The page prints the opener first; the restored form must read the same way."""
        section = Section(
            title="Validation du protocole",
            start_page=1,
            level=2,
            relevance=SectionRelevance.RELEVANT,
            section_number="8.7",
        )

        assert section.printed_heading == "8.7 Validation du protocole"

    def test__a_section_without_an_opener_prints_its_title_alone(self) -> None:
        """A heading that opens straight on its description restores to just that."""
        section = Section(
            title="ANNEXES", start_page=1, level=1, relevance=SectionRelevance.IRRELEVANT
        )

        assert section.printed_heading == "ANNEXES"

    def test__a_code_recorded_in_both_parts_prints_once(self) -> None:
        """The page prints « Lot numéro 146867 » once, so the restored form must too."""
        section = Section(
            title="Lot numéro 146867 - Essais de charge",
            start_page=2,
            level=2,
            relevance=SectionRelevance.RELEVANT,
            section_number="3.3 Lot numéro 146867",
        )

        assert section.printed_heading == "3.3 Lot numéro 146867 - Essais de charge"

    def test__the_overlap_is_read_under_the_comparable_form(self) -> None:
        """A cased or accented duplication is still one phrase, never two."""
        section = Section(
            title="LOT NUMÉRO 146867 - Essais de charge",
            start_page=2,
            level=2,
            relevance=SectionRelevance.RELEVANT,
            section_number="3.3 Lot numero 146867",
        )

        assert section.printed_heading == "3.3 Lot numero 146867 - Essais de charge"


class TestLocatedSectionBody:
    """``body`` is the text without its own printed heading."""

    @staticmethod
    def located(title: str, text: str) -> LocatedSection:
        """The section under test — only the title and the text matter here."""
        return LocatedSection(
            title=title, start_page=1, level=2, relevance=SectionRelevance.RELEVANT, text=text
        )

    def test__the_leading_title_is_stripped_from_the_text(self) -> None:
        """The heading is already a field of its own; the body must not repeat it."""
        section = self.located("ESSAIS DE CHARGE", "ESSAIS DE CHARGE\nLes essais ont")

        assert section.body == "Les essais ont"

    def test__the_title_is_stripped_despite_accents_and_spacing(self) -> None:
        """The printed body renders the title with its own spacing and line breaks."""
        section = self.located("Essais de charge", "Essais   de charge — Les essais ont")

        assert section.body == "Les essais ont"

    def test__a_text_that_does_not_open_on_its_title_is_untouched(self) -> None:
        """Stripping only removes a leading heading, never body text."""
        section = self.located("ESSAIS DE CHARGE", "Les essais ont")

        assert section.body == "Les essais ont"

    def test__a_heading_only_section_has_an_empty_body(self) -> None:
        """A rubric prints nothing under its heading."""
        section = self.located("Table des matières", "Table des matières")

        assert not section.body

    def test__a_section_the_locator_found_no_text_for_has_an_empty_body(self) -> None:
        """No text found means nothing to extract, not a heading to re-read."""
        assert not self.located("Rapport du comité technique", "").body


class TestFileReference:
    ANNOT: ClassVar[BBox] = BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
    """Where the annotation sits — irrelevant to the reference, required by the model."""

    SUFFIXES: ClassVar[list[str]] = [".pdf", ".xlsx"]
    """The formats under test: a PDF and a workbook."""

    def test__filename_is_the_targets_last_segment(self) -> None:
        """A URL and a GoToR path both end on the file they name."""
        url = Link(target="https://example.org/docs/RT-2025-245.pdf", bbox=self.ANNOT)
        path = Link(target="annexes/plan.pdf", bbox=self.ANNOT)

        references = FileReference.from_links([url, path], self.SUFFIXES)

        assert [reference.filename for reference in references] == ["RT-2025-245.pdf", "plan.pdf"]

    def test__a_workbook_is_a_document_and_a_web_page_is_not(self) -> None:
        """An Excel link is kept; a site link, a bare separator and a folder are left out."""
        links = [
            Link(target="https://example.org/", bbox=self.ANNOT),
            Link(target="U:/finances/Sommaire_des_imprevus.xlsx", bbox=self.ANNOT),
            Link(target="/", bbox=self.ANNOT),
            Link(target="https://example.org/docs/", bbox=self.ANNOT),
        ]

        references = FileReference.from_links(links, self.SUFFIXES)

        assert [r.target for r in references] == ["U:/finances/Sommaire_des_imprevus.xlsx"]

    def test__a_format_outside_the_accepted_ones_is_left_out(self) -> None:
        """The suffixes given decide: the same workbook link is dropped when only PDF is kept."""
        link = Link(target="U:/finances/Sommaire_des_imprevus.xlsx", bbox=self.ANNOT)

        assert FileReference.from_links([link], [".pdf"]) == []

    def test__the_same_document_linked_twice_is_one_reference(self) -> None:
        """A code printed twice in the margin points once at its file."""
        link = Link(target="https://example.org/CV-2025-1367.pdf", bbox=self.ANNOT)

        assert len(FileReference.from_links([link, link], self.SUFFIXES)) == 1


class TestResolvedSection:
    class Inspection(SectionEntities):
        """A project's extraction schema: the base mentions plus a field of its own."""

        summary: str = ""

    def test__a_project_schema_survives_in_the_record_and_its_dump(self) -> None:
        """The section keeps the subclass instance, and its dump keeps the subclass's fields."""
        entities = self.Inspection(
            summary="Essais conformes",
            mentions=[Mention(kind="contract", value="C-12", role=MentionRole.OWN)],
        )

        section = ResolvedSection(
            title="Essais",
            start_page=1,
            level=1,
            relevance=SectionRelevance.RELEVANT,
            entities=entities,
        )

        assert section.entities is entities
        assert section.model_dump()["entities"]["summary"] == "Essais conformes"

    def test__an_unread_section_carries_empty_entities(self) -> None:
        """A section nobody read still answers its mentions, empty."""
        section = ResolvedSection(
            title="Annexes", start_page=1, level=1, relevance=SectionRelevance.IRRELEVANT
        )

        assert section.entities.mentions == []
        assert section.extraction_error is None
