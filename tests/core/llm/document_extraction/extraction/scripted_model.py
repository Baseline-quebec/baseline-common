"""ScriptedModel — models that answer a fixed object, or must never be called, for agent tests."""

from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


class ScriptedModel:
    """Builds function models standing in for a server: no network, every prompt recorded."""

    @staticmethod
    def answering(args: dict[str, Any], seen: list[str] | None = None) -> FunctionModel:
        """Return a model that answers ``args`` through the output tool every time.

        Args:
            args: The object the model answers.
            seen: Collects the last part of every request the model was sent, when given.

        Returns:
            The model.
        """

        def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if seen is not None:
                request = next(m for m in reversed(messages) if isinstance(m, ModelRequest))
                part = request.parts[-1]
                assert isinstance(part, UserPromptPart)
                seen.append(str(part.content))

            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)]
            )

        return FunctionModel(answer)

    @staticmethod
    def silent() -> FunctionModel:
        """Return a model that fails the test if it is ever called.

        Returns:
            The model.
        """

        def refuse(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            message = "the model must not be called in this test"
            raise AssertionError(message)

        return FunctionModel(refuse)
