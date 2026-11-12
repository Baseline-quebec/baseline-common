"""Unit tests for WindowCarry: what a reconciled window hands the next one."""

from baseline_core.core.llm.document_extraction.indexing.section_reconciler.window_carry import (
    WindowCarry,
)
from baseline_core.core.llm.document_extraction.section import Section, SectionRelevance


class CarryCase:
    """Builds the kept sections a carry is read from."""

    @staticmethod
    def kept(title: str, page: int, level: int, section_number: str | None = None) -> Section:
        """Build one kept section.

        Args:
            title: Its heading.
            page: Its 1-based start page.
            level: Its depth in the plan.
            section_number: The opener its title was split from, if any.

        Returns:
            The section.
        """
        return Section(
            title=title,
            start_page=page,
            level=level,
            relevance=SectionRelevance.IRRELEVANT,
            section_number=section_number,
        )


class TestFromReconciled(CarryCase):
    """The open plan is the chain of ancestors of the last section kept, and nothing closed."""

    def test__the_open_plan_is_the_ancestor_chain_of_the_last_section(self) -> None:
        """A closed rubric and a closed sibling are left out, and so is the last section itself."""
        kept = [
            self.kept("FINANCES", 3, 1),
            self.kept("Budget", 4, 2),
            self.kept("MAINTENANCE", 120, 1),
            self.kept("Pump A", 150, 2),
            self.kept("Pump B", 190, 2),
            self.kept("Seals", 198, 3),
        ]

        carry = WindowCarry.from_reconciled(200, [], kept)

        assert [(s.title, s.level) for s in carry.open_plan] == [("MAINTENANCE", 1), ("Pump B", 2)]

    def test__a_flat_plan_leaves_nothing_open(self) -> None:
        """Every section at level 1 means no section contains what follows the seam."""
        kept = [self.kept("A", 1, 1), self.kept("B", 2, 1)]

        assert WindowCarry.from_reconciled(200, [], kept).open_plan == []

    def test__nothing_kept_leaves_nothing_open(self) -> None:
        """A first window that kept no section still hands its conventions on."""
        carry = WindowCarry.from_reconciled(200, ["rubrics are CAPS"], [])

        assert carry.open_plan == []
        assert carry.conventions == ["rubrics are CAPS"]


class TestText(CarryCase):
    """The carry reads as a preamble: pages done, conventions, then the open plan."""

    def test__renders_conventions_and_the_open_plan_with_their_levels(self) -> None:
        """The next window sees each open section with its level, page and printed heading."""
        kept = [self.kept("FINANCES", 3, 1, "4."), self.kept("Budget 2026", 198, 2)]

        text = WindowCarry.from_reconciled(200, ["rubrics are CAPS"], kept).text

        assert text.startswith("Pages 1 to 200 are already reconciled.\n")
        assert "- rubrics are CAPS\n" in text
        assert '- L1 p3 "4. FINANCES"\n' in text
        assert text.endswith('- L2 p198 "Budget 2026"')
        # The open ancestor comes before the closed last section, each under its own heading.
        assert text.index("run on past") < text.index("The last section reported")

    def test__a_flat_plan_carries_its_last_section_as_closed_and_nothing_open(self) -> None:
        """The level anchor survives on a flat plan, without any section being called open."""
        text = WindowCarry.from_reconciled(200, [], [self.kept("A", 1, 1)]).text

        assert "run on past" not in text
        assert text.endswith(
            "The last section reported, which the next heading of its rank closes — the first "
            "headings of this window are its peers unless printed as its items:\n"
            '- L1 p1 "A"'
        )

    def test__an_empty_carry_still_says_which_pages_are_done(self) -> None:
        """Even with nothing to hand on, the next window is told where the document stands."""
        text = WindowCarry.from_reconciled(200, [], []).text

        assert text == "Pages 1 to 200 are already reconciled."
