"""What identifies a section's own item, and how wide the repair looks at a boundary."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.section_entities import Mention, MentionRole


class SectionRepairConfig(BaseModel):
    """The project's identity kinds and the repair's thresholds, shared by detector and agent."""

    model_config = ConfigDict(frozen=True)

    identity_kinds: Annotated[
        list[str],
        Field(
            description=(
                "The mention kinds that identify the item a section is about — a contract "
                "number, a work-order code, a report reference. Two sections OWNING the same "
                "one may be one item; each owning a distinct one are two items. Every other "
                "kind (a site, a supplier, a standard) is only a weak signal."
            )
        ),
    ]
    min_shared_signals: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Weak signals two neighbours must share to reach the agent. No weak signal "
                "suffices alone; two agreeing sit badly on chance inside one document."
            ),
        ),
    ] = 2
    boundary_chars: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "Characters kept on each side of the boundary; 600 covers a whole paragraph, "
                "which is where the evidence of a split lives."
            ),
        ),
    ] = 600

    def identifies(self, mention: Mention) -> bool:
        """Whether a mention is the section's own identity: owned, and of an identity kind.

        Args:
            mention: One mention of a section.

        Returns:
            True for an identity the section owns; a cited code never identifies it.
        """
        return mention.role is MentionRole.OWN and mention.kind in self.identity_kinds
