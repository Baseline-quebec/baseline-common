"""RecordMatchingAgent: decides which known record a written record is, if any."""

from pydantic_ai import Agent
from pydantic_ai.models import Model

from baseline_core.service.record_matching.agent.deps import RecordMatchingDeps
from baseline_core.service.record_matching.agent.output import MatchDecision
from baseline_core.service.record_matching.agent.prompt import RecordMatchingPrompt


class RecordMatchingAgent(Agent[RecordMatchingDeps, MatchDecision]):
    """Decides which candidate a written record is, or that it is none of them."""

    def __init__(self, model: Model, *, instructions: str | None = None) -> None:
        """Bind the model; the output type and the system prompt are the agent's own.

        Args:
            model: The LLM model to use.
            instructions: System prompt override; defaults to ``RecordMatchingPrompt.SYSTEM``.
        """
        super().__init__(
            model=model,
            deps_type=RecordMatchingDeps,
            output_type=MatchDecision,
            instructions=instructions or RecordMatchingPrompt.SYSTEM,
        )

    async def decide(self, deps: RecordMatchingDeps) -> MatchDecision:
        """Decide which candidate the written record is.

        Args:
            deps: The written record and the candidates near it.

        Returns:
            The verdict: the candidate's id with a match; the reason and a leaning when uncertain.
        """
        result = await self.run(self.user_prompt(deps), deps=deps)

        return result.output

    @staticmethod
    def user_prompt(deps: RecordMatchingDeps) -> str:
        """Render the written record, then the candidates.

        Args:
            deps: The written record and the candidates near it.

        Returns:
            The user prompt.
        """
        written = deps.written
        rendered = [
            "<written>",
            f"    name: {written.name}",
            f"    code: {written.code or '-'}",
            f"    address: {written.address or '-'}",
            "</written>",
            "",
            "<candidates>",
        ]

        for candidate in deps.candidates:
            rendered += [
                f'    <record id="{candidate.id_}">',
                f"        name: {candidate.name}",
                f"        address: {candidate.address or '-'}",
                "    </record>",
            ]

        return "\n".join([*rendered, "</candidates>"])
