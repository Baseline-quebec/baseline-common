"""A project's extraction schema, for the tests: the base mentions and dates, plus a summary."""

from baseline_core.core.llm.document_extraction.section_entities import SectionEntities


class ReportEntities(SectionEntities):
    """The base mentions and dates, plus a field of the project's own."""

    summary: str | None = None
