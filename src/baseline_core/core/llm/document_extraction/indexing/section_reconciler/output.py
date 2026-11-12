"""The reconciler's output envelope: the shape the model answers in, never a domain record."""

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.section import Section


class TableOfContent(BaseModel):
    """The sections the reconciler kept, and the reading it kept them under."""

    model_config = ConfigDict(frozen=True)

    conventions: list[str] = Field(
        default=[],
        description=(
            "FIRST, one short line each: the rules THIS document prints, read off the candidates "
            "you were given. How a rubric is set apart from an item, what marks depth, which "
            "repeated lines are furniture or form labels. Settle them once here, then apply them "
            "to every section below."
        ),
    )
    """Scratchpad the next window of a long document is handed; removing it changes the answers."""

    sections: list[Section] = Field(..., title="Kept sections")
