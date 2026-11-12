"""When a search stops: an iteration cap, a time budget, a stagnation window, or any of them."""

import time
from typing import ClassVar


class MaxIterations:
    """Stop once ``limit`` iterations have run."""

    def __init__(self, limit: int) -> None:
        """Keep the cap.

        Args:
            limit: The iterations a search may run.

        Raises:
            ValueError: The cap is not positive.
        """
        if limit <= 0:
            msg = f"an iteration cap must be positive; got {limit}"
            raise ValueError(msg)

        self.limit = limit

    def should_stop(self, iteration: int) -> bool:
        """Whether the cap is reached before ``iteration`` runs.

        Args:
            iteration: The iteration about to run, from 0.

        Returns:
            ``iteration >= limit``.
        """
        return iteration >= self.limit

    def checkpoint(self, iteration: int) -> int:
        """The cap itself: nothing changes before it.

        Args:
            iteration: The iteration just asked about.

        Returns:
            The cap.
        """
        del iteration
        return self.limit

    @staticmethod
    def record_progress(*, improved_best: bool) -> None:
        """Ignore the outcome: only the count decides.

        Args:
            improved_best: Whether the iteration beat the best.
        """
        del improved_best


class MaxTime:
    """Stop once a wall-clock budget, started at the first question, has run out."""

    CLOCK_STRIDE: ClassVar[int] = 4096
    """Iterations between two looks at the clock, for a loop too hot to look at every one."""

    def __init__(self, budget_seconds: float) -> None:
        """Keep the budget; the clock starts at the first ``should_stop``.

        Args:
            budget_seconds: The seconds the search may take.

        Raises:
            ValueError: The budget is not positive.
        """
        if budget_seconds <= 0:
            msg = f"a time budget must be positive; got {budget_seconds}"
            raise ValueError(msg)

        self.budget_seconds = budget_seconds
        self._deadline: float | None = None

    def should_stop(self, iteration: int) -> bool:
        """Whether the deadline has passed; the first call sets it.

        Args:
            iteration: The iteration about to run.

        Returns:
            Whether the clock is past the deadline.
        """
        del iteration
        now = time.monotonic()

        if self._deadline is None:
            self._deadline = now + self.budget_seconds
            return False

        return now >= self._deadline

    def checkpoint(self, iteration: int) -> int:
        """``CLOCK_STRIDE`` iterations later: the clock may pass at any time.

        Args:
            iteration: The iteration just asked about.

        Returns:
            The next iteration to look at the clock.
        """
        return iteration + self.CLOCK_STRIDE

    @staticmethod
    def record_progress(*, improved_best: bool) -> None:
        """Ignore the outcome: only the clock decides.

        Args:
            improved_best: Whether the iteration beat the best.
        """
        del improved_best


class NoImprovement:
    """Stop after ``patience`` iterations in a row that do not beat the best."""

    def __init__(self, patience: int) -> None:
        """Keep the window.

        Args:
            patience: The non-improving iterations in a row tolerated.

        Raises:
            ValueError: The window is not positive.
        """
        if patience <= 0:
            msg = f"a patience must be positive; got {patience}"
            raise ValueError(msg)

        self.patience = patience
        self._since_improvement = 0

    def should_stop(self, iteration: int) -> bool:
        """Whether the non-improving streak has reached the patience.

        Args:
            iteration: The iteration about to run.

        Returns:
            Whether ``patience`` iterations in a row failed to improve.
        """
        del iteration
        return self._since_improvement >= self.patience

    def checkpoint(self, iteration: int) -> int:
        """Where the patience runs out, were nothing to improve until then.

        Args:
            iteration: The iteration just asked about.

        Returns:
            The earliest iteration the patience is spent at.
        """
        return iteration + max(self.patience - self._since_improvement, 1)

    def record_progress(self, *, improved_best: bool) -> None:
        """Restart the streak on an improvement, else extend it.

        Args:
            improved_best: Whether the iteration beat the best.
        """
        self._since_improvement = 0 if improved_best else self._since_improvement + 1


class AnyOf:
    """Stop as soon as one of its conditions stops."""

    def __init__(self, conditions: "list[StopCondition]") -> None:
        """Keep the conditions.

        Args:
            conditions: At least one condition.

        Raises:
            ValueError: No condition is given.
        """
        if not conditions:
            msg = "AnyOf needs at least one condition"
            raise ValueError(msg)

        self.conditions = conditions

    def should_stop(self, iteration: int) -> bool:
        """Whether any condition stops.

        Args:
            iteration: The iteration about to run.

        Returns:
            Whether one of them stops.
        """
        return any(condition.should_stop(iteration) for condition in self.conditions)

    def checkpoint(self, iteration: int) -> int:
        """The earliest checkpoint of its conditions.

        Args:
            iteration: The iteration just asked about.

        Returns:
            The earliest iteration one of them must be asked at.
        """
        return min(condition.checkpoint(iteration) for condition in self.conditions)

    def record_progress(self, *, improved_best: bool) -> None:
        """Pass the outcome to every condition.

        Args:
            improved_best: Whether the iteration beat the best.
        """
        for condition in self.conditions:
            condition.record_progress(improved_best=improved_best)


type StopCondition = MaxIterations | MaxTime | NoImprovement | AnyOf
