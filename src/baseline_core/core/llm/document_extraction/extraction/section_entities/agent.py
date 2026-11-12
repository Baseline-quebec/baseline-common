"""SectionEntitiesAgent — reads what one section names off its text, in the project's schema."""

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models import Model

from baseline_core.core.llm.document_extraction.extraction.mention_harvester import Candidate
from baseline_core.core.llm.document_extraction.extraction.section_entities.config import (
    SectionEntitiesConfig,
)
from baseline_core.core.llm.document_extraction.extraction.section_entities.prompt import (
    SectionEntitiesPrompt,
)
from baseline_core.core.llm.document_extraction.section import LocatedSection
from baseline_core.core.llm.document_extraction.section_entities import SectionEntities


class SectionEntitiesAgent[Out: SectionEntities](Agent[None, Out]):
    """Reads the mentions and the project's own fields of one section, off its text."""

    def __init__(
        self, model: Model, output_type: type[Out], config: SectionEntitiesConfig
    ) -> None:
        """Initialize the agent.

        Args:
            model: The LLM model to use.
            output_type: The project's extraction schema, a subclass of ``SectionEntities``.
            config: The project's mention kinds, and how much of a section is sent.
        """
        # No function tool: otherwise a model now and then answered nothing, or an undeclared tool.
        super().__init__(
            model=model,
            output_type=output_type,
            instructions=SectionEntitiesPrompt.system(config.kinds),
            model_settings=ModelSettings(tool_choice="none"),
        )
        self._config = config

    async def extract(self, section: LocatedSection, candidates: list[Candidate]) -> Out:
        """Read what the section names.

        Args:
            section: The section, its heading split from its body.
            candidates: The harvested values, in printed order; the only values of their kinds.

        Returns:
            The reading — an empty object for a section that names nothing.
        """
        prompt = SectionEntitiesPrompt.user(
            section, candidates[: self._config.candidate_limit], self._config.body_char_limit
        )
        result = await self.run(prompt)
        return result.output
