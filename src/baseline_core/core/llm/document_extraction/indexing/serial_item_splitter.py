"""SerialItemSplitter — a section that swallowed items of its series is split at their headings."""

import re
from typing import ClassVar

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    SerialCode,
    SerialItem,
)
from baseline_core.core.llm.document_extraction.section import (
    LocatedSection,
    Section,
    SectionRelevance,
)


class SerialItemSplitter:
    """Splits a section at each promoted item heading its text opens a line with, nothing more."""

    CLOSING_LINE: ClassVar[re.Pattern[str]] = re.compile(r"\d{1,3}(?:\.\d{1,3})+\s+\d{6,}")
    """An item number and a file code alone (« 40.04 1247905019 »): closes an item, opens none."""

    NON_WORD: ClassVar[re.Pattern[str]] = re.compile(r"\W+")
    """What two printings of one heading may differ by: spacing and punctuation."""

    DEFAULT_LEVEL: ClassVar[int] = 2
    """The level an inserted item takes when no section the reconciler kept heads an item."""

    @classmethod
    def split(cls, located: list[LocatedSection], items: list[SerialItem]) -> list[Section]:
        """Insert, after each section, the items whose headings its text opens.

        Args:
            located: The located sections, in the reconciler's order.
            items: Each serial item the finder promoted.

        Returns:
            The sections to locate again, each swallowed item inserted after its section.
        """
        by_code = {
            item.code: item.candidate
            for item in items
            if not cls.CLOSING_LINE.fullmatch(item.candidate.text.strip())
        }
        headed = {
            code.code
            for section in located
            for text in [section.printed_heading, section.title]
            if (code := SerialCode.opening(text)) is not None
        }
        level = next(
            (
                section.level
                for section in located
                if (code := SerialCode.opening(section.printed_heading)) is not None
                and code.code in by_code
            ),
            cls.DEFAULT_LEVEL,
        )
        sections: list[Section] = []

        for section in located:
            sections.append(
                Section.model_validate(section.model_dump(include=set(Section.model_fields)))
            )

            for line in section.text.splitlines()[1:]:
                code = SerialCode.opening(line)
                candidate = by_code.get(code.code) if code is not None else None

                if code is None or candidate is None or code.code in headed:
                    continue

                if candidate.page < section.start_page:
                    continue

                printed = cls.NON_WORD.sub("", line).lower()
                promoted = cls.NON_WORD.sub("", candidate.text).lower()

                # The code opens a different line than the promoted heading: a citation.
                if not (promoted.startswith(printed) or printed.startswith(promoted)):
                    continue

                headed.add(code.code)
                sections.append(
                    Section(
                        title=candidate.text.strip(),
                        start_page=candidate.page,
                        level=level,
                        relevance=SectionRelevance.UNCERTAIN,
                    )
                )

        return sections
