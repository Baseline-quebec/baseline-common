"""Unit tests for LlmUsage: a run's tokens summed, then priced in CAD at Azure's list price."""

from decimal import Decimal

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from baseline_core.core.llm.usage import LlmUsage


class TestCost:
    """Cached input is billed at its own price; reasoning is in the output tokens."""

    def test__the_tokens__are_priced_by_kind(self) -> None:
        """gpt-5-4: 800 000 input at 3.8955, 200 000 cached at 0.3896, 100 000 out at 23.3731."""
        usage = LlmUsage(
            model="gpt-5-4", input_tokens=1_000_000, cached_tokens=200_000, output_tokens=100_000
        )

        assert usage.cost_cad == Decimal("5.531630")

    def test__a_model_without_a_price__has_no_cost(self) -> None:
        """Its tokens still count."""
        usage = LlmUsage(model="test", input_tokens=10, cached_tokens=0, output_tokens=5)

        assert (usage.cost_cad, usage.output_tokens) == (None, 5)


class TestOf:  # codespell:ignore
    """What every response of a run spent, summed under the agent's deployment."""

    def test__the_responses_of_a_run__are_summed(self) -> None:
        """Two responses, a retry say; the requests carry no usage."""
        agent = Agent(
            FunctionModel(lambda *_: ModelResponse(parts=[]), model_name="gpt-5-6-terra")
        )
        answer = ModelResponse(
            parts=[TextPart("{}")],
            usage=RequestUsage(input_tokens=1_000, cache_read_tokens=400, output_tokens=50),
        )
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("…")]), answer, answer]

        usage = LlmUsage.of(agent, messages)

        assert usage == LlmUsage(
            model="gpt-5-6-terra", input_tokens=2_000, cached_tokens=800, output_tokens=100
        )

    def test__a_run_the_model_never_answered__spent_nothing(self) -> None:
        """No response, no usage."""
        agent = Agent(FunctionModel(lambda *_: ModelResponse(parts=[]), model_name="gpt-5-4"))

        assert LlmUsage.of(agent, [ModelRequest(parts=[UserPromptPart("…")])]) is None
