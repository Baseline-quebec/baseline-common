"""Doubles and fixtures for the eval-kit tests: a scored system under test, in miniature."""

import asyncio
import os
from collections.abc import Mapping
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from baseline_core.core.llm.evalkit.abstract_eval_adapter import AbstractEvalAdapter
from baseline_core.core.llm.evalkit.abstract_gated_score import AbstractGatedScore
from baseline_core.core.llm.evalkit.abstract_metric import AbstractMetric, MetricResult
from baseline_core.core.llm.evalkit.case import EvalCase, EvalDataset, EvalOutput
from baseline_core.core.llm.evalkit.evaluator import EvalResult


def pytest_configure() -> None:
    """Silence MLflow's coding-agent hint before a test module imports MLflow."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


class ItemRecall(BaseModel):
    """Rich detail payload of :class:`ItemRecallMetric`, standing in for a real score."""

    model_config = ConfigDict(frozen=True)

    hits: int
    missed: list[str]


class ItemRecallMetric(AbstractMetric[list[str], list[str]]):
    """Test double: the share of expected items the output contains."""

    PASS_MARK: ClassVar[float] = 0.9
    """Threshold the doubles are gated at -- the kit requires one, it defaults to none."""

    @staticmethod
    def make_result(
        name: str,
        expected: list[str],
        output: list[str],
        **cost: Any,  # noqa: ANN401
    ) -> EvalResult[list[str], list[str], list[str]]:
        """Build one result scored by this metric, without going through the run loop.

        Args:
            name: The case name.
            expected: The labeled items.
            output: The observed items.
            **cost: Optional ``input_tokens``/``output_tokens``/``elapsed_s`` of the envelope.

        Returns:
            A result carrying a single ``ItemRecallMetric`` score.
        """
        case = EvalCase[list[str], list[str]](name=name, input=expected, expected=expected)
        metric = ItemRecallMetric(ItemRecallMetric.PASS_MARK)
        return EvalResult(
            case=case,
            output=EvalOutput(value=output, **cost),
            scores={"ItemRecallMetric": metric.score(expected, output)},
        )

    def score(self, expected: list[str], output: list[str]) -> MetricResult[ItemRecall]:
        """Score the overlap and report what was missed.

        Args:
            expected: The labeled items.
            output: The observed items.

        Returns:
            The overlap share, with the missed items as typed detail.
        """
        missed = [item for item in expected if item not in output]
        hits = len(expected) - len(missed)
        share = hits / len(expected) if expected else 1.0
        return MetricResult(
            score=share,
            reason=f"{hits}/{len(expected)} matched",
            detail=ItemRecall(hits=hits, missed=missed),
        )


class ExactMatchMetric(AbstractMetric[list[str], list[str]]):
    """Test double: 1.0 only when the output equals the expected list."""

    display_name = "Exactness"

    def score(self, expected: list[str], output: list[str]) -> MetricResult[None]:
        """Score exact equality of the two lists.

        Args:
            expected: The labeled items.
            output: The observed items.

        Returns:
            One when the lists are equal, zero otherwise.
        """
        equal = expected == output
        return MetricResult(score=1.0 if equal else 0.0, reason="equal" if equal else "differs")


class InputCopyingAdapter(AbstractEvalAdapter[list[str], list[str]]):
    """Test double: a deterministic system that echoes its input, reporting no cost."""

    def __init__(self, drop: int = 0) -> None:
        """Bind how many trailing items the echo drops.

        Args:
            drop: How many trailing items the system fails to emit.
        """
        self.seen: list[list[str]] = []
        self._drop = drop

    async def evaluate(self, case_input: list[str]) -> EvalOutput[list[str]]:
        """Echo the input minus the dropped tail.

        Args:
            case_input: The case input.

        Returns:
            The echoed output, with every cost field left unset.
        """
        self.seen.append(case_input)
        kept = case_input[: len(case_input) - self._drop] if self._drop else case_input
        return EvalOutput(value=kept)


class TokenBillingAdapter(AbstractEvalAdapter[list[str], list[str]]):
    """Test double: a system that bills its calls, like an LLM-backed one."""

    async def evaluate(self, case_input: list[str]) -> EvalOutput[list[str]]:
        """Echo the input and bill one token per item.

        Args:
            case_input: The case input.

        Returns:
            The echoed output with its token counts and wall time.
        """
        return EvalOutput(
            value=case_input,
            input_tokens=len(case_input),
            output_tokens=2 * len(case_input),
            elapsed_s=0.5,
        )


class InFlightCountingAdapter(AbstractEvalAdapter[list[str], list[str]]):
    """Test double: a system that records how many of its calls were in flight at once."""

    def __init__(self) -> None:
        """Start with nothing in flight."""
        self.in_flight = 0
        self.peak = 0

    async def evaluate(self, case_input: list[str]) -> EvalOutput[list[str]]:
        """Echo the input, yielding once so a sibling case can start before this one ends.

        Args:
            case_input: The case input.

        Returns:
            The echoed output.
        """
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return EvalOutput(value=case_input)


class FailingAdapter(AbstractEvalAdapter[list[str], list[str]]):
    """Test double: a system that fails, the way a rate-limited one does mid-run."""

    def __init__(self, only: str | None = None) -> None:
        """Bind the cases the system fails on.

        Args:
            only: Fail solely on the cases whose input holds this item; ``None`` fails them all.
        """
        self._only = only

    async def evaluate(self, case_input: list[str]) -> EvalOutput[list[str]]:
        """Raise on the selected cases and echo the rest.

        Args:
            case_input: The case input.

        Returns:
            The echoed output, for a case this double spares.

        Raises:
            RuntimeError: When this case is one the double is set to fail.
        """
        if self._only is None or self._only in case_input:
            msg = "adapter exploded"
            raise RuntimeError(msg)

        return EvalOutput(value=case_input)


class DetectionCounts(BaseModel, AbstractGatedScore):
    """Stand-in for a project's rich per-case score, gated on derived properties."""

    model_config = ConfigDict(frozen=True)

    hits: int
    detected: int
    expected: int

    def gated_numbers(self) -> Mapping[str, float]:
        """Name the numbers this double exposes to a gate.

        Returns:
            Its precision, recall and coverage, unrounded.
        """
        return {"precision": self.precision, "recall": self.recall, "coverage": self.coverage}

    @property
    def precision(self) -> float:
        """Share of detected items that are real."""
        return self.hits / self.detected if self.detected else 1.0

    @property
    def recall(self) -> float:
        """Share of expected items that were detected."""
        return self.hits / self.expected if self.expected else 1.0

    @property
    def coverage(self) -> float:
        """Share of expected items captured by anything at all."""
        return self.recall


@pytest.fixture
def dataset() -> EvalDataset[list[str], list[str]]:
    """A two-case dataset whose input doubles as its expected output."""
    return EvalDataset(
        name="mini_corpus",
        cases=[
            EvalCase(name="first", input=["a", "b"], expected=["a", "b"]),
            EvalCase(name="second", input=["c"], expected=["c"]),
        ],
    )


@pytest.fixture
def cheap_run() -> list[EvalResult[list[str], list[str], list[str]]]:
    """Return two results, one of which reports no cost at all."""
    return [
        ItemRecallMetric.make_result(
            "first", ["a"], ["a"], input_tokens=10, output_tokens=4, elapsed_s=1.5
        ),
        ItemRecallMetric.make_result(
            "second", ["b"], [], input_tokens=6, output_tokens=2, elapsed_s=0.5
        ),
    ]
