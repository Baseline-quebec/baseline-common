"""The engine's solver, for any model: a bound model and a start in, a solution out."""

from numpy.random import default_rng
from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.model_solution import ModelSolution, VariableAssignment
from baseline_core.core.optimization.solver.search.destroy_repair import DestroyRepair, Release
from baseline_core.core.optimization.solver.search.destroy_selector import (
    AlnsConfig,
    DestroySelector,
)
from baseline_core.core.optimization.solver.search.kernel_evaluator import KernelEvaluator
from baseline_core.core.optimization.solver.search.kernel_state import KernelState
from baseline_core.core.optimization.solver.search.search_index import SearchIndex
from baseline_core.core.optimization.solver.stop_condition import (
    AnyOf,
    MaxIterations,
    MaxTime,
    NoImprovement,
    StopCondition,
)


class KernelSolverConfig(BaseModel):
    """The settings of a solve: the destroy arms, the budget and the seed."""

    model_config = ConfigDict(frozen=True)

    alns: AlnsConfig = Field(default_factory=AlnsConfig)
    """How the destroy arms are weighted."""
    n_iters: int = Field(default=2_000, gt=0)
    """Iteration cap; one iteration is one destroy and repair."""
    patience: int = Field(default=400, gt=0)
    """Consecutive non-improving iterations tolerated before the search stops."""
    time_budget_seconds: float | None = Field(default=None, gt=0.0)
    """Seconds the search takes at most; ``None`` stops on iterations and patience alone."""
    seed: int = 42
    """The seed of every random draw: the same seed, the same search."""


class KernelSolver:
    """A destroy-and-repair search on the kernel of any bound model, keeping moves no worse."""

    # no compiler here: ``solve`` takes a bound model, so one bind serves a pre-check and solves
    def __init__(self, config: KernelSolverConfig) -> None:
        """Keep the solve's settings.

        Args:
            config: The destroy arms, the budget and the seed.
        """
        self.config = config

    def solve(
        self,
        bound: BoundModel,
        warm_start: list[VariableAssignment] | None = None,
        fixed: list[VariableAssignment] | None = None,
    ) -> ModelSolution:
        """Place the warm start and the fixes, search, leave out what still breaks a hard rule.

        Args:
            bound: The model bound to the problem.
            warm_start: Where the search begins; an assignment the model cannot read is skipped.
            fixed: What the search may not change; a ``None`` value or target stays free.

        Returns:
            The best assignments reached, the elements left out, the kernel's counts and scores.
        """
        config = self.config
        bound.place(warm_start or [])
        index = SearchIndex.of(bound, fixed)
        kernel = bound.kernel
        state = KernelState(kernel)
        self._hold_fixed(state, index)

        # soft from the start: a hard-only first pass leaves soft costs that are never undone
        kernel.soft_mode = True
        rng = default_rng(config.seed)

        if not warm_start:
            DestroyRepair(Release(list(range(kernel.n)), remove_first=True), index, rng).apply(
                state
            )

        selector = DestroySelector(index, config.alns, rng, config.seed)
        self._search(state, selector, KernelEvaluator(bound), self._stop)
        self._leave_out_conflicts(state, index)

        return bound.solution

    @staticmethod
    def _search(
        state: KernelState,
        selector: DestroySelector,
        evaluator: KernelEvaluator,
        stop: StopCondition,
    ) -> None:
        """Destroy and repair until the stop, then go back to the best placement reached.

        Args:
            state: The placement, edited in place.
            selector: Draws each iteration's move.
            evaluator: Scores each move against the running score.
            stop: When the search ends.
        """
        best_snapshot, best = state.snapshot(), evaluator.current
        iteration = 0

        while not stop.should_stop(iteration):
            iteration += 1
            current = evaluator.current
            move = selector.select(state)

            if move is None:
                # a draw that released nothing still costs its arm, but not the patience
                selector.feedback(improved_best=False, improved_current=False, accepted=False)
                continue

            change = evaluator.apply(move, state)
            candidate = current + change

            accepted = candidate <= current
            improved_best = accepted and candidate < best

            if accepted:
                evaluator.commit(change)

                if improved_best:
                    best_snapshot, best = state.snapshot(), candidate
            else:
                move.revert(state)

            selector.feedback(
                improved_best=improved_best,
                improved_current=candidate < current,
                accepted=accepted,
            )
            stop.record_progress(improved_best=improved_best)

        state.restore(best_snapshot)

    @property
    def _stop(self) -> StopCondition:
        """The search's stop: iterations, stagnation, and the time budget when one is set."""
        config = self.config
        conditions: list[StopCondition] = [
            MaxIterations(config.n_iters),
            NoImprovement(config.patience),
        ]

        if config.time_budget_seconds is not None:
            conditions.append(MaxTime(config.time_budget_seconds))

        return AnyOf(conditions)

    @staticmethod
    def _hold_fixed(state: KernelState, index: SearchIndex) -> None:
        """Move each fixed element to the one start, and into the one set, its domain has left.

        Args:
            state: The placement, edited in place.
            index: The domains, cut to the fixed values.
        """
        kernel = state.kernel
        starts, targets = kernel.placement

        for element in range(kernel.n):
            if not index.fixed[element]:
                continue

            own_starts, own_sets = index.starts(element), index.sets(element)
            kernel.insert(
                element,
                own_starts[0] if len(own_starts) == 1 else starts[element],
                own_sets[0] if len(own_sets) == 1 else targets[element],
            )

    def _leave_out_conflicts(self, state: KernelState, index: SearchIndex) -> None:
        """Leave out free elements in a hard conflict; a fixed one stays, its conflict counted.

        Args:
            state: The best placement, edited in place.
            index: What stands or falls together, and what is fixed.
        """
        kernel = state.kernel

        while conflicting := [
            j for j in kernel.violating if not any(index.fixed[k] for k in self._unit(index, j))
        ]:
            hardest = max(kernel.element_hard(j) for j in conflicting)
            tied = [j for j in conflicting if kernel.element_hard(j) == hardest]
            softs = [self._soft_without(state, index, j) for j in tied]
            chosen = tied[softs.index(min(softs))]

            for k in self._unit(index, chosen):
                kernel.remove(k)

    def _soft_without(self, state: KernelState, index: SearchIndex, element: int) -> int:
        """The soft score once an element's unit is out, the placement left as it was.

        Args:
            state: The placement.
            index: What stands or falls together.
            element: The element tried out.

        Returns:
            The weighted soft score without its unit.
        """
        kernel = state.kernel
        snapshot = state.snapshot()

        for k in self._unit(index, element):
            kernel.remove(k)

        soft: int = kernel.full_soft
        state.restore(snapshot)
        return soft

    @staticmethod
    def _unit(index: SearchIndex, element: int) -> list[int]:
        """The element with what stands or falls with it: its groups and its partner, closed.

        Args:
            index: What stands or falls together.
            element: The element.

        Returns:
            Those elements, ascending.
        """
        unit, stack = {element}, [element]

        while stack:
            k = stack.pop()
            linked = {m for group in index.groups if k in group for m in group}

            if index.partner[k] >= 0:
                linked.add(index.partner[k])

            for m in linked - unit:
                unit.add(m)
                stack.append(m)

        return sorted(unit)
