"""What the relevance fallback looks for, and how much text under a heading it reads."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SectionRelevanceConfig(BaseModel):
    """The project's relevance criterion, and the window the fallback reads and slides."""

    model_config = ConfigDict(frozen=True)

    relevance_criterion: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "What makes a section relevant to the project, in plain words: what it must "
                "concern, and what it must name for that to count — e.g. « a repair or a "
                "replacement of equipment at an identified site ». The same criterion the "
                "reconciler judged the heading by."
            ),
        ),
    ]
    window_chars: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "Characters of printed text one window carries, sized to hold a whole ordinary "
                "section: a section cut in two hands the model its preamble without the "
                "conclusion it leads to, and the preamble alone reads as something else."
            ),
        ),
    ] = 2000
    overlap_chars: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Characters the next window re-reads, so a passage the cut fell inside is still "
                "read whole by one of them."
            ),
        ),
    ] = 300
    max_windows: Annotated[
        int, Field(gt=0, description="Windows read at most before the section is left uncertain.")
    ] = 3

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        """Validate that the overlap is smaller than the window it re-reads.

        Returns:
            The config, unchanged.

        Raises:
            ValueError: When the overlap would keep the window from moving.
        """
        if self.overlap_chars >= self.window_chars:
            msg = (
                f"overlap_chars ({self.overlap_chars}) must be < "
                f"window_chars ({self.window_chars})"
            )
            raise ValueError(msg)

        return self
