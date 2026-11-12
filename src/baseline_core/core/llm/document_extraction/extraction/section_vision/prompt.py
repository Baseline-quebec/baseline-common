"""The vision agent's prompts: the text agent's rules, over the pages a section prints."""

import inspect
from typing import ClassVar

from baseline_core.core.llm.document_extraction.extraction.section_entities.prompt import (
    SectionEntitiesPrompt,
)


class SectionVisionPrompt:
    """Renders the static system prompt of a project and the user prompt of one section."""

    TASK: ClassVar[str] = inspect.cleandoc(
        """
        <task>
            You extract what ONE section of a document names, from the images of the pages
            it prints: scans, drawings, photographed forms. You receive its heading and its
            pages — never the rest of the document. Report only what THESE pages show of this
            section. The description of each field of the output schema is the project's
            instruction for that field: follow it.
            Pages that name nothing yield an empty object, and that is a correct answer.
        </task>
        """
    )

    PIXELS: ClassVar[str] = inspect.cleandoc(
        """
        <reading_pixels>
            Read every value off the image exactly as printed, character by character. A
            value you cannot read with certainty — blurred, cut by the page edge, half
            hidden by a stamp — is left out: a misread code is a false join between two
            unrelated matters, worse than a code missed. A title block, a stamp or a form
            field names the section's own subject; a list of reference drawings or documents
            names what it cites.
        </reading_pixels>
        """
    )

    @classmethod
    def system(cls, kinds: dict[str, str]) -> str:
        """Render the system prompt of a project, the same for every section.

        Args:
            kinds: Each mention kind of the project, with a one-line description.

        Returns:
            The system prompt.
        """
        return "\n\n".join([
            cls.TASK,
            SectionEntitiesPrompt.kinds_block(kinds),
            SectionEntitiesPrompt.RULES,
            cls.PIXELS,
        ])

    @staticmethod
    def user(title: str) -> str:
        """Render the text that opens the section's images.

        Args:
            title: The section's heading, as printed.

        Returns:
            The text part of the user prompt.
        """
        return (
            f"<section_title>\n{title}\n</section_title>\n\n"
            "The pages of this section follow as images."
        )
