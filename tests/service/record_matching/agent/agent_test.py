import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from baseline_core.service.record_matching.agent.agent import RecordMatchingAgent
from baseline_core.service.record_matching.agent.deps import RecordMatchingDeps
from baseline_core.service.record_matching.agent.output import MatchDecision, MatchVerdict
from baseline_core.service.record_matching.agent.prompt import RecordMatchingPrompt


class ScriptedModel:
    """A model that answers one decision and keeps the prompts it was sent."""

    def __init__(self, answer: dict[str, str]) -> None:
        """Keep the answer to give.

        Args:
            answer: The decision's fields, as the model would write them.
        """
        self.answer = answer
        self.prompts: list[str] = []
        self.instructions: list[str | None] = []

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Record the user prompt, then call the output tool with the scripted decision.

        Args:
            messages: The conversation so far.
            info: The run's tools, among them the output tool.

        Returns:
            The call to the output tool.
        """
        self.prompts += [
            part.content
            for message in messages
            for part in message.parts
            if part.part_kind == "user-prompt" and isinstance(part.content, str)
        ]
        self.instructions += [getattr(message, "instructions", None) for message in messages[-1:]]
        tool = info.output_tools[0].name

        return ModelResponse(parts=[ToolCallPart(tool_name=tool, args=self.answer)])


class TestRecordMatchingAgent:
    async def test__decide__returns_the_model_s_decision(self, deps: RecordMatchingDeps) -> None:
        """A match names the candidate's id."""
        model = ScriptedModel({"verdict": "match", "record_id": "org-1"})
        agent = RecordMatchingAgent(FunctionModel(model.respond))

        decision = await agent.decide(deps)

        assert decision == MatchDecision(verdict=MatchVerdict.MATCH, record_id="org-1")

    async def test__the_prompt__shows_the_written_record_and_every_candidate(
        self, deps: RecordMatchingDeps
    ) -> None:
        """The id is for the verdict, the name and address for the evidence."""
        model = ScriptedModel({"verdict": "new_record"})
        agent = RecordMatchingAgent(FunctionModel(model.respond))

        await agent.decide(deps)

        prompt = model.prompts[0]
        assert "name: ACME IND. (MTL)" in prompt
        assert "code: C-0042" in prompt
        assert '<record id="org-1">' in prompt
        assert '<record id="org-2">' in prompt

    @pytest.mark.parametrize(
        ("override", "expected"), [(None, RecordMatchingPrompt.SYSTEM), ("Decide.", "Decide.")]
    )
    async def test__the_system_prompt__is_the_agent_s_own_unless_overridden(
        self, deps: RecordMatchingDeps, override: str | None, expected: str
    ) -> None:
        """An override replaces the instructions the model receives; nothing else changes."""
        model = ScriptedModel({"verdict": "new_record"})
        agent = RecordMatchingAgent(FunctionModel(model.respond), instructions=override)

        await agent.decide(deps)

        assert model.instructions == [expected]
