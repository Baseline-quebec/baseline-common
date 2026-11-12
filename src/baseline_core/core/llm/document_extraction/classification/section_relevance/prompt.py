"""The relevance fallback's prompt: the project's criterion, read against printed text."""

import inspect
from typing import ClassVar

from baseline_core.core.llm.document_extraction.section import LocatedSection


class SectionRelevancePrompt:
    """Renders the fallback's instructions around a criterion, and one window as its message."""

    SYSTEM: ClassVar[str] = inspect.cleandoc(
        """
        <task>
            The reconciliation step could not settle one section's relevance from its
            heading alone. You are given that heading and a window of the text printed
            under it. Answer with the section's relevance, and nothing else.
        </task>

        <relevance_criterion>
            {relevance_criterion}
        </relevance_criterion>

        <relevance_rules>
            Does the section concern what the criterion above names? When the
            criterion asks for several things together — what is done AND what it is
            done to, say — every one of them must be there.
            - relevant: everything the criterion asks for is present in the text,
              whatever kind of act carries it, and whether that act approves it,
              refuses it or only reports on it.
            - irrelevant: the text plainly concerns something the criterion leaves
              out.
            - uncertain: one part of the criterion is there and another is not, or
              this window does not say. A heading with almost no text under it is
              uncertain, never irrelevant: nothing was read that could rule it out.
            In doubt, uncertain — NEVER relevant.
        </relevance_rules>
        """
    )

    @classmethod
    def system(cls, relevance_criterion: str) -> str:
        """Render the instructions around the project's criterion.

        Args:
            relevance_criterion: What makes a section relevant, as the project words it.

        Returns:
            The system prompt.
        """
        return cls.SYSTEM.format(relevance_criterion=relevance_criterion.strip())

    @staticmethod
    def user(section: LocatedSection, window: str) -> str:
        """Render one window as the agent's user message.

        Args:
            section: The uncertain section under test.
            window: One window of the text printed under its heading.

        Returns:
            The heading as the page prints it, then that text, verbatim.
        """
        # The printed heading, opener included: relevance is judged on what detection surfaced.
        return f'Section heading: "{section.printed_heading}"\n\nText printed under it:\n{window}'
