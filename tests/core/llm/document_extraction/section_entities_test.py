"""Tests of the mention envelope and the deterministic merge of two readings."""

from datetime import date

from pydantic import Field

from baseline_core.core.llm.document_extraction.section_entities import (
    Mention,
    MentionRole,
    SectionEntities,
)


class TestMention:
    def test__key__ignores_punctuation_spacing_accents_and_case(self) -> None:
        """Two printings of one code meet on one key."""
        printed = Mention(kind="contract", value="CÉ-2026 / 0145", role=MentionRole.OWN)
        reprinted = Mention(kind="contract", value="ce20260145", role=MentionRole.CITED)

        assert printed.key == reprinted.key == "ce20260145"


class TestSectionEntitiesMergedWith:
    class Inspection(SectionEntities):
        """A project's extraction schema with a scalar and a list of its own."""

        summary: str = ""
        findings: list[str] = Field(default_factory=list)

    def test__lists_unite_on_their_keys_and_this_reading_wins_a_mention(self) -> None:
        """A mention both readings name keeps this reading's printing and role."""
        text = self.Inspection(
            mentions=[Mention(kind="contract", value="C-12", role=MentionRole.OWN)],
            findings=["fissure"],
        )
        vision = self.Inspection(
            mentions=[
                Mention(kind="contract", value="c 12", role=MentionRole.CITED),
                Mention(kind="site", value="Bâtiment B", role=MentionRole.OWN),
            ],
            findings=["fissure", "corrosion"],
        )

        merged = text.merged_with(vision)

        assert [mention.value for mention in merged.mentions] == ["C-12", "Bâtiment B"]
        assert merged.mentions[0].role is MentionRole.OWN
        assert merged.findings == ["fissure", "corrosion"]

    def test__a_scalar_is_taken_from_this_reading_first(self) -> None:
        """The other reading fills only what this one left empty."""
        text = self.Inspection(summary="Essais conformes")
        vision = self.Inspection(summary="Autre lecture", dated=date(2026, 3, 2))

        merged = text.merged_with(vision)

        assert merged.summary == "Essais conformes"
        assert merged.dated == date(2026, 3, 2)

    def test__the_merge_keeps_the_projects_schema(self) -> None:
        """Merging two readings of a subclass gives that subclass back."""
        merged = self.Inspection().merged_with(self.Inspection(summary="x"))

        assert isinstance(merged, self.Inspection)
