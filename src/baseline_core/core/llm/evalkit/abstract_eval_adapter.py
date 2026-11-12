"""The contract a system under test wears to be evaluated by the kit."""

from abc import ABC, abstractmethod

from baseline_core.core.llm.evalkit.case import EvalOutput


class AbstractEvalAdapter[I, O](ABC):
    """Adapts the system under test to the kit's uniform evaluate interface."""

    @abstractmethod
    async def evaluate(self, case_input: I) -> EvalOutput[O]:
        """Run the system on one case input.

        Args:
            case_input: The case input.

        Returns:
            The observed output, wrapped with what producing it cost.
        """
