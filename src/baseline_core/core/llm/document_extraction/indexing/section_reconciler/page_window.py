"""PageWindow — the slice of a document one call reads, and the pages it answers for."""

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    HeadingCandidates,
)
from baseline_core.core.llm.document_extraction.section import Section


class PageWindow(BaseModel):
    """A run of consecutive pages whose sections one call reports, cut after detection."""

    model_config = ConfigDict(frozen=True)

    first_page: int = Field(..., ge=1, description="First page the window answers for.")
    last_page: int = Field(..., ge=1, description="Last page the window answers for, inclusive.")
    rank: int = Field(..., ge=1, description="Position of the window in its document, from 1.")
    count: int = Field(..., ge=1, description="How many windows the document was cut into.")
    headings: HeadingCandidates = Field(..., description="What both sources surfaced on the run.")

    @property
    def partial(self) -> bool:
        """Whether the window is one slice of a longer document rather than the whole of it."""
        return self.count > 1

    def owns(self, section: Section) -> bool:
        """Tell whether a reported section opens on a page this window answers for.

        Args:
            section: A section the model reported for this window.

        Returns:
            True when its start page sits in the run, False when a neighbour owns it.
        """
        # Open-ended at the document's edges: a page past the last candidate is no one else's.
        after_start = self.rank == 1 or section.start_page >= self.first_page
        before_end = self.rank == self.count or section.start_page <= self.last_page
        return after_start and before_end

    @classmethod
    def tile(cls, headings: HeadingCandidates, size: int) -> list["PageWindow"]:
        """Cut one document's candidates into consecutive windows of ``size`` pages.

        Args:
            headings: Both detection sources of the whole document.
            size: Pages each window answers for; the last one takes what remains.

        Returns:
            The windows in page order; none when nothing was surfaced.
        """
        last = headings.last_page
        starts = range(1, last + 1, size)
        return [
            cls(
                first_page=start,
                last_page=min(start + size - 1, last),
                rank=rank,
                count=len(starts),
                headings=headings.between(start, start + size - 1),
            )
            for rank, start in enumerate(starts, start=1)
        ]
