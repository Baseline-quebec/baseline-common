"""Tests of the vision agent: the pages reach the model as images, the schema comes back."""

import asyncio
from io import BytesIO

from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from baseline_core.core.llm.document_extraction.extraction.section_entities.prompt import (
    SectionEntitiesPrompt,
)
from baseline_core.core.llm.document_extraction.extraction.section_vision.agent import (
    SectionVisionAgent,
)
from baseline_core.core.llm.document_extraction.extraction.section_vision.config import (
    SectionVisionConfig,
)
from tests.core.llm.document_extraction.extraction.report_entities import ReportEntities
from tests.core.llm.document_extraction.extraction.scripted_model import ScriptedModel


class TestSectionVisionAgent:
    """One call per batch: the heading as text, then one image per page."""

    def test__the_pages_reach_the_model_as_images_after_the_heading(self) -> None:
        """The model is sent the heading, then each page as a PNG, in order."""
        sent: list[list[object]] = []
        instructions: list[str | None] = []

        def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            request = next(m for m in reversed(messages) if isinstance(m, ModelRequest))
            part = request.parts[-1]
            assert isinstance(part, UserPromptPart)
            sent.append(list(part.content))
            instructions.append(info.instructions)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args={"mentions": [{"kind": "site", "value": "Pier 4", "role": "own"}]},
                    )
                ]
            )

        out = BytesIO()
        Image.new("RGB", (4, 4)).save(out, format="PNG")
        agent = SectionVisionAgent(
            FunctionModel(answer), ReportEntities, SectionVisionConfig(kinds=ReportEntities.KINDS)
        )

        entities = asyncio.run(agent.extract("Site photographs", [out.getvalue()] * 2))

        assert isinstance(entities, ReportEntities)
        assert [m.value for m in entities.mentions] == ["Pier 4"]
        assert "<section_title>\nSite photographs\n</section_title>" in str(sent[0][0])
        assert [type(part) for part in sent[0][1:]] == [BinaryContent, BinaryContent]
        assert instructions[0] is not None
        assert SectionEntitiesPrompt.RULES in instructions[0]

    def test__the_render_resolution_is_the_configured_one(self) -> None:
        """The service renders at the agent's resolution, measured with its model."""
        config = SectionVisionConfig(kinds=ReportEntities.KINDS, dpi=200)

        agent = SectionVisionAgent(ScriptedModel.silent(), ReportEntities, config)

        assert agent.dpi == 200
