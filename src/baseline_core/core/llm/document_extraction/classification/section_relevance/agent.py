"""SectionRelevanceAgent — the fallback settling an uncertain section on the text under it."""

from pydantic_ai import Agent
from pydantic_ai.models import Model

from baseline_core.core.llm.document_extraction.classification.section_relevance.config import (
    SectionRelevanceConfig,
)
from baseline_core.core.llm.document_extraction.classification.section_relevance.prompt import (
    SectionRelevancePrompt,
)
from baseline_core.core.llm.document_extraction.section import LocatedSection, SectionRelevance


class SectionRelevanceAgent(Agent[None, SectionRelevance]):
    """Settles a section the reconciler left uncertain by reading the text printed under it."""

    def __init__(self, model: Model, *, config: SectionRelevanceConfig) -> None:
        """Initialize the agent.

        Args:
            model: The LLM model to use.
            config: The project's criterion, and how much text the windows carry.
        """
        super().__init__(
            model=model,
            output_type=SectionRelevance,
            instructions=SectionRelevancePrompt.system(config.relevance_criterion),
        )
        self._config = config

    async def settle(self, section: LocatedSection) -> SectionRelevance:
        """Slide a window down the text under a heading until one window settles the section.

        Args:
            section: The section the reconciler left uncertain, with its own text.

        Returns:
            The first relevance a window settles on, else ``uncertain``.
        """
        for window in self._windows(section.text):
            result = await self.run(SectionRelevancePrompt.user(section, window))

            if result.output is not SectionRelevance.UNCERTAIN:
                return result.output

        return SectionRelevance.UNCERTAIN

    def _windows(self, text: str) -> list[str]:
        """Cut the text under a heading into the overlapping windows this agent reads.

        Args:
            text: The text printed under the heading, as the locator cut it.

        Returns:
            The windows in reading order; empty when the section carries no text.
        """
        size, limit = self._config.window_chars, self._config.max_windows
        step = max(1, size - self._config.overlap_chars)
        windows: list[str] = []
        start = 0

        while start < len(text) and len(windows) < limit:
            windows.append(text[start : start + size])

            # Sliding past the end would only re-read a sliver of the tail.
            if start + size >= len(text):
                break

            start += step

        return windows
