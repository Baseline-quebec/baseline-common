"""The text agent's knobs: the project's mention kinds, and how much of a section is sent."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class SectionEntitiesConfig(BaseModel):
    """What the text agent tells the model, and how much of the section reaches the prompt."""

    model_config = ConfigDict(frozen=True)

    kinds: dict[str, str] = Field(
        ...,
        min_length=1,
        description=(
            "Each mention kind of the project, with a one-line description: the only kinds "
            "the model may give a mention (« supplier »: « an organization that delivers »)."
        ),
    )
    body_char_limit: Annotated[
        int,
        Field(
            gt=0,
            description="Characters kept of the section body; the cap only guards against "
            "runaway appendices.",
        ),
    ] = 12_000
    candidate_limit: Annotated[
        int,
        Field(
            gt=0,
            description="Harvested candidates listed to the model; a section printing more "
            "is a list, not a section. Only the prompt is capped: the restriction to the "
            "harvest uses every candidate.",
        ),
    ] = 60
