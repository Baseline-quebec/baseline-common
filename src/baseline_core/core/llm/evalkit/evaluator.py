"""Evaluate a dataset through an adapter, score every case, and hand back typed results."""

import asyncio
from collections.abc import Sequence
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from baseline_core.core.llm.evalkit.abstract_eval_adapter import AbstractEvalAdapter
from baseline_core.core.llm.evalkit.abstract_metric import AbstractMetric, MetricResult
from baseline_core.core.llm.evalkit.case import EvalCase, EvalDataset, EvalOutput


class EvalResult[I, E, O](BaseModel):
    """One case's outcome: the observed output and each metric's score."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    case: EvalCase[I, E]
    output: EvalOutput[O]
    scores: dict[str, MetricResult[Any]]
    """Each metric's score, keyed by the metric's class name."""


class Evaluator[I, E, O]:
    """Evaluates labeled datasets through one system under test and scores what comes back."""

    def __init__(
        self,
        adapter: AbstractEvalAdapter[I, O],
        metrics: Sequence[AbstractMetric[E, O]],
        *,
        tolerate_errors: bool = False,
    ) -> None:
        """Bind the system under test, the metrics scoring it, and how a failing case is treated.

        Args:
            adapter: The system under test, adapted to the kit interface.
            metrics: The metrics scoring each output against its expected value.
            tolerate_errors: Whether one case's failure lets the others still be reported.

        Raises:
            ValueError: When two metrics share a class name, which keys their scores.
        """
        names = [type(metric).__name__ for metric in metrics]

        if repeated := sorted({name for name in names if names.count(name) > 1}):
            msg = f"metrics of the same class would overwrite each other: {', '.join(repeated)}"
            raise ValueError(msg)

        self._adapter = adapter
        self._metrics = metrics
        self._tolerate_errors = tolerate_errors

    @property
    def metrics(self) -> Sequence[AbstractMetric[E, O]]:
        """The metrics every case is scored by."""
        return self._metrics

    def evaluate(self, dataset: EvalDataset[I, E]) -> list[EvalResult[I, E, O]]:
        """Evaluate every case of a dataset through the adapter and score it.

        Args:
            dataset: The named collection of labeled cases.

        Returns:
            One result per case that succeeded, in dataset order.

        Raises:
            RuntimeError: When an event loop is already running in this thread.
        """  # noqa: DOC502 - raised by ``asyncio.run``
        return asyncio.run(self._evaluate_all(dataset))

    async def _evaluate_all(self, dataset: EvalDataset[I, E]) -> list[EvalResult[I, E, O]]:
        """Fan every case out at once and keep the ones that came back.

        Args:
            dataset: The named collection of labeled cases.

        Returns:
            One result per case that succeeded, in dataset order.
        """
        done = await asyncio.gather(*(self._evaluate_case(case) for case in dataset.cases))
        return [result for result in done if result is not None]

    async def _evaluate_case(self, case: EvalCase[I, E]) -> EvalResult[I, E, O] | None:
        """Evaluate and score one case.

        Args:
            case: The labeled case to evaluate.

        Returns:
            Its scored result, or ``None`` when it failed and errors are tolerated.
        """
        try:
            output = await self._adapter.evaluate(case.input)
        except Exception:
            if not self._tolerate_errors:
                raise

            logger.exception("case {} failed and was dropped from the evaluation", case.name)
            return None

        scores = {
            type(metric).__name__: metric.score(case.expected, output.value)
            for metric in self._metrics
        }
        return EvalResult(case=case, output=output, scores=scores)
