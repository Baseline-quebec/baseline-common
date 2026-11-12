"""Deterministic scoring of an observed output against its labeled truth."""

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


class MetricResult[D](BaseModel):
    """The score of one metric on one case, plus its metric-specific detail."""

    model_config = ConfigDict(frozen=True)

    score: float
    reason: str
    detail: D | None = None


class AbstractMetric[E, O](ABC):
    """Deterministic scoring of an observed output against the truth (Strategy)."""

    display_name: ClassVar[str | None] = None
    """Label a reporting layer prefers over the class name; derived from it when unset."""

    WORD_BOUNDARY: ClassVar[re.Pattern[str]] = re.compile(r"(?<!^)(?=[A-Z])")
    """Where a class name splits into words, so ``LevelAccuracyMetric`` reads "Level Accuracy"."""

    CLASS_SUFFIX: ClassVar[str] = "Metric"
    """The word every metric class ends on, dropped from the derived label."""

    def __init__(self, threshold: float = 0.0, *, mean: float | None = None) -> None:
        """Bind the gates the metric answers to: per case, and over the sample.

        Args:
            threshold: The lowest score one case may still pass on; 0.0 gates no case.
            mean: The lowest mean the sample may still pass on; ``None`` gates no mean.
        """
        self.threshold = threshold
        self.mean = mean

    @property
    def label(self) -> str:
        """How a reporting layer names this metric, derived from the class name when unset."""
        if self.display_name:
            return self.display_name

        words = self.WORD_BOUNDARY.split(type(self).__name__)
        return " ".join(word for word in words if word != self.CLASS_SUFFIX)

    @abstractmethod
    def score(self, expected: E, output: O) -> MetricResult[Any]:
        """Score one observed output against its expected value.

        Args:
            expected: The labeled ground truth of the case.
            output: What the system under test produced.

        Returns:
            The score and the reason behind it.
        """

    def passes(self, result: MetricResult[Any]) -> bool:
        """Whether a score clears this metric's threshold."""
        return result.score >= self.threshold
