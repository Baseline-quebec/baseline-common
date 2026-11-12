"""Tests of the harvest: what the formats find, and how a reading is held to it."""

import pytest
from pydantic import ValidationError

from baseline_core.core.llm.document_extraction.extraction.mention_harvester import (
    MentionFormat,
    MentionHarvester,
)
from baseline_core.core.llm.document_extraction.section_entities import Mention, MentionRole
from tests.core.llm.document_extraction.extraction.report_entities import ReportEntities


class TestHarvest:
    """Every printed value of a known shape, found whole, in printed order, each once."""

    def test__every_printed_value_is_found_with_its_kind(
        self, harvester: MentionHarvester
    ) -> None:
        """Both printings of the shape come back, typed by their format."""
        found = harvester.harvest("Order REF-2025-777 replaces REF-2024-012.")

        assert [(c.value, c.kind) for c in found] == [
            ("REF-2025-777", "reference"),
            ("REF-2024-012", "reference"),
        ]

    def test__a_value_is_found_whole_never_as_a_tail(self) -> None:
        """« XREF-2025-777 » and « REF-2025-7770 » are other tokens, not this shape."""
        harvester = MentionHarvester([MentionFormat(kind="reference", pattern=r"REF-\d{4}-\d{3}")])

        assert harvester.harvest("XREF-2025-777 and REF-2025-7770") == []

    def test__a_value_printed_twice_is_offered_once(self, harvester: MentionHarvester) -> None:
        """The first printing is kept; the model needs the value, not its count."""
        found = harvester.harvest("REF-2025-777, see ref-2025-777 again")

        assert [c.value for c in found] == ["REF-2025-777"]

    def test__a_named_value_group_keeps_only_the_value(self) -> None:
        """A label may introduce the code: only the code is harvested."""
        harvester = MentionHarvester([
            MentionFormat(kind="notice", pattern=r"notice\s+(?P<value>\d{6,10})")
        ])

        [found] = harvester.harvest("Published under Notice  1908765 last week.")

        assert (found.value, found.kind) == ("1908765", "notice")

    def test__a_code_inside_a_longer_one_is_that_longer_code(self) -> None:
        """« PLAN 2025-03107 » holds « 2025-03107 »: the inner digits are not a second code."""
        harvester = MentionHarvester([
            MentionFormat(kind="drawing", pattern=r"PLAN\s?\d{4}-\d{5}"),
            MentionFormat(kind="order", pattern=r"\d{4}-\d{5}"),
        ])

        found = harvester.harvest("See PLAN 2025-03107 and order 2025-00393.")

        assert [(c.value, c.kind) for c in found] == [
            ("PLAN 2025-03107", "drawing"),
            ("2025-00393", "order"),
        ]

    def test__at_one_printing_the_first_format_decides(self) -> None:
        """Two formats matching the same value: the priority order settles its kind."""
        harvester = MentionHarvester([
            MentionFormat(kind="order", pattern=r"\d{4}-\d{3}"),
            MentionFormat(kind="invoice", pattern=r"\d{4}-\d{3}"),
        ])

        assert [c.kind for c in harvester.harvest("2025-123")] == ["order"]

    def test__a_pattern_that_does_not_compile_is_refused_at_construction(self) -> None:
        """A broken shape fails when the project declares it, never mid-harvest."""
        with pytest.raises(ValidationError, match="does not compile"):
            MentionFormat(kind="reference", pattern=r"REF-(\d{4}")

    def test__an_inline_flag_is_refused_at_construction(self) -> None:
        """The harvester wraps each shape in a group, where a global flag cannot stand."""
        with pytest.raises(ValidationError, match="does not compile"):
            MentionFormat(kind="reference", pattern=r"(?i)ref-\d{4}")


class TestRestrict:
    """The answer's formatted mentions are bounded to the harvest; the others are left alone."""

    PRINTED = "Order REF-2025-777 for the north site, supplied by Acme Tools."

    def reading(self, *mentions: tuple[str, str]) -> ReportEntities:
        """A reading naming each (kind, value), owned."""
        return ReportEntities(
            mentions=[Mention(kind=k, value=v, role=MentionRole.OWN) for k, v in mentions]
        )

    def test__an_invented_code_is_dropped_and_a_harvested_one_retyped(
        self, harvester: MentionHarvester
    ) -> None:
        """The shape decides the kind; a reference the harvest never found does not survive."""
        candidates = harvester.harvest(self.PRINTED)
        reading = self.reading(("site", "REF-2025-777"), ("reference", "REF-2024-999"))

        restricted = harvester.restrict(reading, candidates)

        assert [(m.kind, m.value) for m in restricted.mentions] == [("reference", "REF-2025-777")]

    def test__a_kind_no_format_knows_is_left_as_it_is(self, harvester: MentionHarvester) -> None:
        """A supplier has no shape: the harvest bounds nothing about it."""
        reading = self.reading(("supplier", "Acme Tools"), ("site", "south yard"))

        restricted = harvester.restrict(reading, harvester.harvest(self.PRINTED))

        assert [m.value for m in restricted.mentions] == ["Acme Tools", "south yard"]

    def test__the_project_s_own_fields_survive(self, harvester: MentionHarvester) -> None:
        """Restricting returns the project's schema, its other fields untouched."""
        reading = ReportEntities(summary="A delivery.")

        restricted = harvester.restrict(reading, [])

        assert isinstance(restricted, ReportEntities)
        assert restricted.summary == "A delivery."


class TestDropUnprinted:
    """A text reading keeps only the values its section prints."""

    def test__a_value_the_text_does_not_print_is_dropped(self) -> None:
        """« south yard » is nowhere in the text: the model made it up."""
        reading = ReportEntities(
            mentions=[
                Mention(kind="supplier", value="ACME  tools", role=MentionRole.OWN),
                Mention(kind="site", value="south yard", role=MentionRole.OWN),
            ]
        )

        kept = MentionHarvester.drop_unprinted(reading, "Supplied by Acme Tools, Inc.")

        assert [m.value for m in kept.mentions] == ["ACME  tools"]

    def test__a_value_with_nothing_to_print_is_dropped(self) -> None:
        """A dash names nothing, wherever it prints."""
        reading = ReportEntities(mentions=[Mention(kind="site", value="—", role=MentionRole.OWN)])

        assert MentionHarvester.drop_unprinted(reading, "Site — north").mentions == []

    def test__the_printed_key_reduces_as_a_mention_key_does(self) -> None:
        """The haystack and the needle must share one reduction, accents and case included."""
        value = "Bâtiment Ölund-Straße 12"

        mention = Mention(kind="site", value=value, role=MentionRole.OWN)

        assert Mention.keyed(value) == mention.key
