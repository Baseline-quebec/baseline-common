"""SectionVisionAgent — reads a section's image-heavy pages as pixels, in the project's schema."""

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

from baseline_core.core.llm.document_extraction.extraction.section_vision.config import (
    SectionVisionConfig,
)
from baseline_core.core.llm.document_extraction.extraction.section_vision.prompt import (
    SectionVisionPrompt,
)
from baseline_core.core.llm.document_extraction.section_entities import SectionEntities


class SectionVisionAgent[Out: SectionEntities](Agent[None, Out]):
    """Reads the same schema as the text agent, from the pages a section prints."""

    def __init__(self, model: Model, output_type: type[Out], config: SectionVisionConfig) -> None:
        """Initialize the agent.

        Args:
            model: The LLM model to use — one that accepts image input.
            output_type: The project's extraction schema, the text agent's own.
            config: The project's mention kinds, and the render resolution.
        """
        super().__init__(
            model=model,
            output_type=output_type,
            instructions=SectionVisionPrompt.system(config.kinds),
        )
        self._config = config

    @property
    def dpi(self) -> int:
        """Resolution the caller renders this agent's pages at."""
        return self._config.dpi

    async def extract(self, title: str, pages: list[bytes]) -> Out:
        """Read what the section's pages show.

        Args:
            title: The section's heading, as printed.
            pages: The section's image-heavy pages, one rendered PNG each.

        Returns:
            The reading — an empty object for pages that name nothing.
        """
        prompt: list[str | BinaryContent] = [SectionVisionPrompt.user(title)]
        prompt += [BinaryContent(data=page, media_type="image/png") for page in pages]
        result = await self.run(prompt)
        return result.output
