"""The vision agent's knobs: the project's mention kinds, and the render resolution."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class SectionVisionConfig(BaseModel):
    """What the vision agent tells the model, and how its pages are rendered."""

    model_config = ConfigDict(frozen=True)

    kinds: dict[str, str] = Field(
        ...,
        min_length=1,
        description="Each mention kind of the project, with a one-line description: the "
        "same as the text agent's.",
    )
    dpi: Annotated[
        int,
        Field(
            gt=0,
            description="Render resolution. 150 keeps a letter page under ~1600px on its "
            "long side — enough to read a drawing's title block, unmeasured beyond that.",
        ),
    ] = 150
