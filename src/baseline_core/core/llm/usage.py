"""LlmUsage — the tokens an agent's run spent, and what they cost at Azure's list price, in CAD."""

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model


class TokenPrice(BaseModel):
    """What a million tokens of a model cost, in Canadian dollars."""

    model_config = ConfigDict(frozen=True)

    input: Decimal
    cached_input: Decimal
    output: Decimal


class LlmUsage(BaseModel):
    """The tokens an agent's run spent on its model, and their cost at list price."""

    model_config = ConfigDict(frozen=True)

    PRICES_CAD: ClassVar[dict[str, TokenPrice]] = {
        "gpt-5-4": TokenPrice(
            input=Decimal("3.8955"), cached_input=Decimal("0.3896"), output=Decimal("23.3731")
        ),
        "gpt-5-4-mini": TokenPrice(
            input=Decimal("1.1687"), cached_input=Decimal("0.1169"), output=Decimal("7.0119")
        ),
        "gpt-5-5": TokenPrice(
            input=Decimal("7.791"), cached_input=Decimal("0.7791"), output=Decimal("46.7462")
        ),
        "gpt-5-6-terra": TokenPrice(
            input=Decimal("3.1164"), cached_input=Decimal("0.3116"), output=Decimal("18.6985")
        ),
    }
    """Azure's list price per million tokens by deployment, Data Zone Standard in eastus2, short
    context where it is priced apart (Retail Prices API in CAD, read 2026-10-30)."""

    MILLION: ClassVar[Decimal] = Decimal(1_000_000)
    CENT_FRACTION: ClassVar[Decimal] = Decimal("0.000001")

    model: str = Field(description="The deployment the agent called.")
    input_tokens: int = Field(description="Every input token, those read from the cache included.")
    cached_tokens: int = Field(description="The input tokens read from the cache, billed less.")
    output_tokens: int = Field(description="The output tokens, reasoning included.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_cad(self) -> Decimal | None:
        """What the tokens cost at list price, in Canadian dollars; None for a model not priced."""
        price = self.PRICES_CAD.get(self.model)

        if price is None:
            return None

        cost = (
            (self.input_tokens - self.cached_tokens) * price.input
            + self.cached_tokens * price.cached_input
            + self.output_tokens * price.output
        ) / self.MILLION
        return cost.quantize(self.CENT_FRACTION)

    @classmethod
    def of(cls, agent: Agent[Any, Any], messages: Sequence[ModelMessage]) -> Self | None:
        """Sum what the model answered in a run, as ``capture_run_messages`` caught it.

        Args:
            agent: The agent that ran; its model names the deployment.
            messages: The run's messages.

        Returns:
            The tokens spent; None when the model never answered.
        """
        spent = [message.usage for message in messages if isinstance(message, ModelResponse)]

        if not spent:
            return None

        model = agent.model.model_name if isinstance(agent.model, Model) else str(agent.model)
        return cls(
            model=model,
            input_tokens=sum(usage.input_tokens for usage in spent),
            cached_tokens=sum(usage.cache_read_tokens for usage in spent),
            output_tokens=sum(usage.output_tokens for usage in spent),
        )
