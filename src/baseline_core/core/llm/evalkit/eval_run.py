"""Persist an eval run and total what it cost."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from baseline_core.core.llm.evalkit.case import EvalDataset
from baseline_core.core.llm.evalkit.evaluator import EvalResult


class RunCost(BaseModel):
    """What a whole run cost: tokens and wall time, summed over its cases."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_s: float | None = None

    @classmethod
    def total[I, E, O](cls, results: Sequence[EvalResult[I, E, O]]) -> "RunCost":
        """Total the per-case cost of a run, field by field.

        Args:
            results: The per-case results of one run.

        Returns:
            The totals, each left unset when no case reported that field.
        """
        inputs = [r.output.input_tokens for r in results if r.output.input_tokens is not None]
        outputs = [r.output.output_tokens for r in results if r.output.output_tokens is not None]
        elapsed = [r.output.elapsed_s for r in results if r.output.elapsed_s is not None]
        return cls(
            input_tokens=sum(inputs) if inputs else None,
            output_tokens=sum(outputs) if outputs else None,
            elapsed_s=sum(elapsed) if elapsed else None,
        )

    @property
    def reported(self) -> bool:
        """Whether any case reported a cost at all."""
        return any(
            value is not None for value in (self.input_tokens, self.output_tokens, self.elapsed_s)
        )


class EvalRun[I, E, O, R](BaseModel):
    """One persisted run: the dataset it scored, when, at what cost, with what results."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    ran_at: datetime
    cost: RunCost
    results: list[EvalResult[I, E, O]]
    summary: R | None = None

    @classmethod
    def build(
        cls,
        dataset: EvalDataset[I, E],
        results: Sequence[EvalResult[I, E, O]],
        summary: R | None = None,
    ) -> "EvalRun[I, E, O, R]":
        """Assemble a timestamped run report from per-case results.

        Args:
            dataset: The dataset that was scored; the run is named after it.
            results: The per-case results, in dataset order.
            summary: The corpus-level value folded from those results, when one was computed.

        Returns:
            The run, stamped in UTC, with its cost totals and its optional summary.
        """
        return cls(
            dataset=dataset.name,
            ran_at=datetime.now(UTC),
            cost=RunCost.total(results),
            results=list(results),
            summary=summary,
        )

    def write(self, folder: Path) -> Path:
        """Persist this run as JSON, named after its dataset and its stamp.

        Args:
            folder: The directory holding the run files; created when missing.

        Returns:
            The path written; a run stamped in the same second as another gets a numbered name.
        """
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.dataset}_{self.ran_at:%Y%m%dT%H%M%S}Z"
        text = self.model_dump_json(indent=2) + "\n"
        attempt = 1

        while True:
            path = folder / (f"{stem}.json" if attempt == 1 else f"{stem}_{attempt}.json")

            try:
                with path.open("x", encoding="utf-8") as file:
                    file.write(text)
            except FileExistsError:
                attempt += 1
                continue

            return path
