"""Injectable knobs of the section reconciler: what it judges relevant, how much it reads."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class SectionReconcilerConfig(BaseModel):
    """The project's relevance criterion, and how much of a document one call reads."""

    model_config = ConfigDict(frozen=True)

    relevance_criterion: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "What makes a section relevant to the project, in the project's own words, "
                "injected into the system prompt: « the section concerns maintenance work on "
                "an identified machine or site ». Name every part that must be shown at once "
                "(an act AND what it bears on), and what rules a section out."
            ),
        ),
    ]
    outline_title_limit: Annotated[
        int, Field(gt=0, description="Characters kept of a bookmark title, the rest clipped.")
    ] = 80
    window_pages: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "Pages one call answers for; a longer document is cut into windows of this many "
                "pages, reconciled one after the other, each handed what the last left open, and "
                "stitched by page. A window judges its slice alone, so a document is read whole "
                "whenever it fits."
            ),
        ),
    ] = 400
