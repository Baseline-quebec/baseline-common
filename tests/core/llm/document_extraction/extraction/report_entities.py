"""A project's extraction schema, as the extraction tests read it: one field beside the base."""

from typing import ClassVar

from pydantic import Field

from baseline_core.core.llm.document_extraction.section_entities import SectionEntities


class ReportEntities(SectionEntities):
    """What a section of a technical report names: the base mentions, and a one-line summary."""

    KINDS: ClassVar[dict[str, str]] = {
        "reference": "a report, order or drawing number",
        "site": "a place where the work happens",
        "supplier": "an organization that delivers goods or services",
    }

    summary: str | None = Field(None, description="What the section is about, in one line.")
