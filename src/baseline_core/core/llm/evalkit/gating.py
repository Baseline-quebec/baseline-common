"""Gate a run: per-metric thresholds, a corpus mean, corpus floors, and a frozen reference."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel

from baseline_core.core.llm.evalkit.abstract_gated_score import AbstractGatedScore
from baseline_core.core.llm.evalkit.abstract_metric import AbstractMetric
from baseline_core.core.llm.evalkit.case import EvalDataset
from baseline_core.core.llm.evalkit.evaluator import EvalResult


class Miss(BaseModel):
    """One metric of one case that scored below its own threshold."""

    model_config = ConfigDict(frozen=True)

    case: str
    metric: str
    score: float
    threshold: float

    def __str__(self) -> str:
        """Render how far the metric fell, its case included, on one line.

        Returns:
            The ``case: metric x < threshold y`` clause.
        """
        return f"{self.case}: {self.metric} {self.score:.2f} < threshold {self.threshold:.2f}"


class MetricGate[E, O]:
    """Holds a run to the threshold and the mean each of its metrics carries."""

    def __init__(self, metrics: Sequence[AbstractMetric[E, O]]) -> None:
        """Bind the metrics whose gates hold the run.

        Args:
            metrics: The metrics the run was scored with, each carrying its own gates.
        """
        self._metrics = metrics

    @staticmethod
    def assert_no_case_drops(
        results: Sequence[EvalResult[Any, E, O]], dataset: EvalDataset[Any, E]
    ) -> None:
        """Raise unless every case of the dataset came back with a result.

        Args:
            results: The scored cases of one run.
            dataset: The dataset the run was launched on.

        Raises:
            AssertionError: When at least one case never produced a result.
        """
        ran = {result.case.name for result in results}
        missing = [case.name for case in dataset.cases if case.name not in ran]

        if missing:
            ran_out = f"{len(missing)} of {len(dataset.cases)} cases did not run"
            msg = f"{dataset.name}: {ran_out}: {missing}"
            raise AssertionError(msg)

    def misses(self, results: Sequence[EvalResult[Any, E, O]]) -> list[Miss]:
        """List every metric of every case that came in below its threshold.

        Args:
            results: The scored cases of one run.

        Returns:
            One entry per case and metric that missed, in run then metric order.

        Raises:
            KeyError: When a case was never scored by one of the gated metrics.
        """  # noqa: DOC502 - raised by the scores lookup
        fallen: list[Miss] = []

        for result in results:
            for metric in self._metrics:
                scored = result.scores[type(metric).__name__]

                if not metric.passes(scored):
                    fallen.append(
                        Miss(
                            case=result.case.name,
                            metric=metric.label,
                            score=scored.score,
                            threshold=metric.threshold,
                        )
                    )

        return fallen

    def assert_all_pass(self, results: Sequence[EvalResult[Any, E, O]]) -> None:
        """Raise unless every case cleared every gated metric.

        Args:
            results: The scored cases of one run.

        Raises:
            AssertionError: When no case was scored, or any metric missed its threshold.
        """
        if not results:
            msg = "No cases to gate"
            raise AssertionError(msg)

        fallen = self.misses(results)

        if fallen:
            raise AssertionError("; ".join(str(miss) for miss in fallen))

    def assert_means(self, results: Sequence[EvalResult[Any, E, O]]) -> None:
        """Raise if any metric carrying a mean falls below it over the run.

        Args:
            results: The scored cases of one run.

        Raises:
            AssertionError: When no case was scored, or a mean is below its floor.
            KeyError: When a case was never scored by a mean-gated metric.
        """  # noqa: DOC502 - raised by the scores lookup
        for metric in self._metrics:
            if metric.mean is None:
                continue

            key = type(metric).__name__
            values = {result.case.name: result.scores[key].score for result in results}

            if not values:
                msg = f"No cases to average {metric.label} over"
                raise AssertionError(msg)

            mean = sum(values.values()) / len(values)

            if mean < metric.mean:
                msg = (
                    f"mean {metric.label} {mean:.2f} is below {metric.mean:.2f} over "
                    f"{len(values)} cases: "
                    f"{ {name: round(value, 2) for name, value in sorted(values.items())} }"
                )
                raise AssertionError(msg)


class Regression(BaseModel):
    """One gated field of one case that came in below the frozen reference."""

    model_config = ConfigDict(frozen=True)

    case: str
    field: str
    value: float
    reference: float

    def __str__(self) -> str:
        """Render how far the field fell, its case included, on one line.

        Returns:
            The ``case: field x < reference y`` clause.
        """
        return f"{self.case}: {self.field} {self.value:.2f} < reference {self.reference:.2f}"


class Regressions(RootModel[list[Regression]]):
    """The fallen gated fields of one run, in report order."""

    def __str__(self) -> str:
        """Render every regression as the one-line reason of the assertion that raises them.

        Returns:
            One clause per entry, empty when there is none.
        """
        return "; ".join(str(item) for item in self.root)


class Shortfall(BaseModel):
    """One corpus-level number that came in below its floor."""

    model_config = ConfigDict(frozen=True)

    field: str
    value: float
    floor: float

    def __str__(self) -> str:
        """Render how far the number fell, on one line.

        Returns:
            The ``field x is below y`` clause.
        """
        return f"{self.field} {self.value:.2f} is below {self.floor:.2f}"


class CorpusGate:
    """Holds the corpus-level fold of a run to a floor per gated number."""

    def __init__(self, floors: Mapping[str, float]) -> None:
        """Bind the floor of every gated number.

        Args:
            floors: The minimum each gated number must clear, keyed by the name it reports under.
        """
        self._floors = floors

    def shortfalls(self, scored: AbstractGatedScore) -> list[Shortfall]:
        """List every gated number of a fold that came in below its floor.

        Args:
            scored: The corpus-level fold, which names its own numbers.

        Returns:
            One entry per number that fell, in the order the floors were bound.

        Raises:
            KeyError: When the fold does not name one of the gated numbers.
        """  # noqa: DOC502 - raised by the lookup of a gated number
        reported = scored.gated_numbers()
        return [
            Shortfall(field=field, value=reported[field], floor=floor)
            for field, floor in self._floors.items()
            if reported[field] < floor
        ]

    def assert_clears(self, scored: AbstractGatedScore) -> None:
        """Raise unless every gated number of the fold clears its floor.

        Args:
            scored: The corpus-level fold of one run.

        Raises:
            AssertionError: When at least one gated number is below its floor.
        """
        fallen = self.shortfalls(scored)

        if fallen:
            raise AssertionError("Corpus " + "; ".join(str(short) for short in fallen))


class BaselineGate:
    """Freezes the gated numbers of a run, and holds the next one to that frozen reference."""

    def __init__(self, fields: Sequence[str], digits: int, tol: float) -> None:
        """Bind what is gated, how precisely it is frozen, and how much noise is absorbed.

        Args:
            fields: The gated numbers to read and compare, in report order.
            digits: The decimals kept when freezing a value, so a reference file stays diffable.
            tol: Float tolerance absorbing the rounding noise of a persisted reference.
        """
        self._fields = fields
        self._digits = digits
        self._tol = tol

    def values(self, scored: AbstractGatedScore) -> dict[str, float]:
        """Read and round the gated numbers of one case score.

        Args:
            scored: The score, which names its own numbers.

        Returns:
            The rounded value of each gated field, in the order they were bound.

        Raises:
            KeyError: When the score does not name one of the gated numbers.
        """  # noqa: DOC502 - raised by the lookup of a gated number
        reported = scored.gated_numbers()
        return {name: round(float(reported[name]), self._digits) for name in self._fields}

    def table(self, scores: Mapping[str, AbstractGatedScore]) -> dict[str, dict[str, float]]:
        """Build the per-case table of gated numbers that a reference file freezes.

        Args:
            scores: Per-case scores keyed by case name.

        Returns:
            The gated numbers of every case, keyed by case name.
        """
        return {name: self.values(scored) for name, scored in scores.items()}

    def regressions(
        self,
        current: Mapping[str, Mapping[str, float]],
        reference: Mapping[str, Mapping[str, float]],
    ) -> Regressions:
        """List the gated numbers of a run that came in below a frozen reference.

        Args:
            current: The gated numbers of this run, keyed by case name.
            reference: The frozen reference numbers, keyed by case name.

        Returns:
            One entry per case and field that fell; anything the reference lacks is skipped.
        """
        worse: list[Regression] = []

        for name, values in current.items():
            base = reference.get(name)

            if base is None:
                continue

            for field in self._fields:
                if field not in base or field not in values:
                    continue

                if values[field] < base[field] - self._tol:
                    worse.append(
                        Regression(
                            case=name, field=field, value=values[field], reference=base[field]
                        )
                    )

        return Regressions(worse)
