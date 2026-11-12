"""The labeled case, the dataset collecting them, and the envelope one evaluation produces."""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict


class EvalCase[I, E](BaseModel):
    """A labeled case: the system input and the expected ground truth."""

    model_config = ConfigDict(frozen=True)

    name: str
    input: I
    expected: E


class EvalDataset[I, E](BaseModel):
    """A named collection of labeled cases."""

    model_config = ConfigDict(frozen=True)

    name: str
    cases: list[EvalCase[I, E]]

    def select(self, names: Iterable[str]) -> "EvalDataset[I, E]":
        """Narrow the dataset to the named cases, in the dataset's order.

        Args:
            names: The names of the cases to keep.

        Returns:
            The subset, named after the dataset and its size; the dataset when all are named.

        Raises:
            ValueError: When a name matches no case.
        """
        wanted = set(names)

        if unknown := wanted - {case.name for case in self.cases}:
            msg = f"{self.name} has no case named {', '.join(sorted(unknown))}"
            raise ValueError(msg)

        kept = [case for case in self.cases if case.name in wanted]

        if len(kept) == len(self.cases):
            return self

        return EvalDataset(name=f"{self.name} ({len(kept)} of {len(self.cases)})", cases=kept)


class EvalOutput[O](BaseModel):
    """What one case produced: the output plus what it cost to produce it."""

    model_config = ConfigDict(frozen=True)

    value: O
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_s: float | None = None
