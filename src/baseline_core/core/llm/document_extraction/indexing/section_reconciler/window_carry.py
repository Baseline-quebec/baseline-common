"""WindowCarry — what one window hands the next: its conventions, and the plan it left open."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.section import Section


class WindowCarry(BaseModel):
    """What a reconciled window hands the next one, so the plan does not restart at the seam."""

    model_config = ConfigDict(frozen=True)

    last_page: int = Field(..., ge=1, description="The last page already reconciled.")
    conventions: list[str] = Field(..., description="The rules the previous window settled.")
    open_plan: list[Section] = Field(
        ..., description="The sections whose contents run on past the seam, outermost first."
    )
    last: Section | None = Field(
        ...,
        description="The last section kept: the level the next window's first ones compare to.",
    )

    @classmethod
    def from_reconciled(cls, last_page: int, conventions: list[str], kept: list[Section]) -> Self:
        """Read what the windows reconciled so far leave open for the next one.

        Args:
            last_page: The last page the previous window answered for.
            conventions: The rules that window read off its candidates.
            kept: Every section kept so far, in document order.

        Returns:
            The carry: the conventions, and the ancestors of the last section kept.
        """
        # The last section is carried apart, as closed: told it was still open, the model folded
        # the next window's banners into it.
        chain: list[Section] = []
        ceiling = kept[-1].level if kept else 0

        for section in reversed(kept):
            if section.level < ceiling:
                chain.append(section)
                ceiling = section.level

        chain.reverse()
        return cls(
            last_page=last_page,
            conventions=conventions,
            open_plan=chain,
            last=kept[-1] if kept else None,
        )

    @property
    def text(self) -> str:
        """The carry as the next window reads it, before its own page blocks."""
        lines = [f"Pages 1 to {self.last_page} are already reconciled."]

        if self.conventions:
            lines.append("Conventions settled on them, to apply unchanged here:")
            lines.extend(f"- {convention}" for convention in self.conventions)

        if self.open_plan:
            lines.append(
                f"The sections whose contents run on past page {self.last_page}, outermost "
                "first — an item under the innermost one is one level below it, and a new "
                "heading of the same rank as one of them is its peer:"
            )
            lines.extend(self._line(section) for section in self.open_plan)

        if self.last is not None:
            lines.extend([
                (
                    "The last section reported, which the next heading of its rank closes — "
                    "the first headings of this window are its peers unless printed as its "
                    "items:"
                ),
                self._line(self.last),
            ])

        return "\n".join(lines)

    @staticmethod
    def _line(section: Section) -> str:
        """Render one carried section as the next window reads it.

        Args:
            section: A kept section.

        Returns:
            Its level, page and printed heading on one line.
        """
        return f'- L{section.level} p{section.start_page} "{section.printed_heading}"'
