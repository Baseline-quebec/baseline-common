import pytest

from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.infeasibility.infeasibility_check import (
    InfeasibilityCheck,
    PrecheckConfig,
)
from baseline_core.core.optimization.solver.output_mapper import OutputMapper
from baseline_core.core.optimization.solver.search.kernel_solver import (
    KernelSolver,
    KernelSolverConfig,
)
from baseline_core.core.repository.abstract_repository import RecordNotFoundError
from baseline_core.core.repository.in_memory_repository import InMemoryRepository
from baseline_core.orchestrator.planning_orchestrator import Plan, PlanningOrchestrator
from baseline_core.service.optimization.optimization_service import OptimizationService
from tests.core.optimization.job_shop_model import JobShopModelDefinition, Schedule, Shop


class StoredShop(Shop):
    """A job shop as a repository holds it: under its id."""

    id: str

    @classmethod
    def of_rows(cls, id_: str, operations: list[list[object]]) -> "StoredShop":
        """A stored shop on M1 and M2, one ``[ref, job, step, duration, machines, release, due]``.

        Args:
            id_: Its id.
            operations: One row per operation.

        Returns:
            The stored shop.
        """
        shop = Shop.of(operations)
        return cls(id=id_, machines=shop.machines, operations=shop.operations)


@pytest.fixture(scope="module")
def service() -> OptimizationService[Shop, Schedule]:
    """The job shop's service: the pre-check under a budget, a short search.

    Returns:
        The service.
    """
    compiled = ModelCompiler(JobShopModelDefinition(), cythonize=False)
    precheck = InfeasibilityCheck(PrecheckConfig(time_budget_seconds=20))
    solver = KernelSolver(KernelSolverConfig(n_iters=200, patience=100))
    return OptimizationService(compiled, precheck, solver, OutputMapper(Schedule))


@pytest.fixture
def problems() -> InMemoryRepository[Shop]:
    """One shop of two jobs that fit on their machines, held with its id ``shop-1``.

    Returns:
        The problems.
    """
    shop = StoredShop.of_rows(
        "shop-1",
        [
            ["a1", "A", 1, 4, ["M1"], 0, 20],
            ["a2", "A", 2, 3, ["M2"], 0, 20],
            ["b1", "B", 1, 5, ["M2"], 0, 20],
            ["b2", "B", 2, 2, ["M1"], 0, 20],
        ],
    )
    return InMemoryRepository(Shop, [shop])


@pytest.fixture
def plans() -> InMemoryRepository[Plan[Schedule]]:
    """No plan yet; one store read and written, as one backend would be.

    Returns:
        The plans.
    """
    return InMemoryRepository(Plan[Schedule])


@pytest.fixture
def orchestrator(
    problems: InMemoryRepository[Shop],
    service: OptimizationService[Shop, Schedule],
    plans: InMemoryRepository[Plan[Schedule]],
) -> PlanningOrchestrator[Shop, Schedule]:
    """The planning flow over the stores.

    Args:
        problems: The problems.
        service: The job shop's service.
        plans: The plans.

    Returns:
        The orchestrator.
    """
    return PlanningOrchestrator[Shop, Schedule](problems, service, plans, plans)


class TestPlanningOrchestrator:
    @staticmethod
    def _placements(plan: Plan[Schedule]) -> dict[str, list[int | str]]:
        """Each scheduled operation's start and machine, by its ref.

        Args:
            plan: A plan.

        Returns:
            ``[start, machine]`` per operation.
        """
        return {op.ref: [op.start, op.machine] for op in plan.result.output.operations}

    async def test__a_plan__is_written_under_the_problem_id(
        self,
        orchestrator: PlanningOrchestrator[Shop, Schedule],
        plans: InMemoryRepository[Plan[Schedule]],
    ) -> None:
        """The plan returned is the one held under the problem's id, every operation placed."""
        plan = await orchestrator.plan("shop-1", check=True)

        assert await plans.get_by_id("shop-1") == plan
        assert plan.result.precheck_issues is not None
        assert plan.result.solution.summary.score.hard == 0
        assert sorted(self._placements(plan)) == ["a1", "a2", "b1", "b2"]

    async def test__a_missing_problem__raises_and_writes_nothing(
        self,
        orchestrator: PlanningOrchestrator[Shop, Schedule],
        plans: InMemoryRepository[Plan[Schedule]],
    ) -> None:
        """A problem id the repository does not hold is refused before any solve."""
        with pytest.raises(RecordNotFoundError, match="shop-2"):
            await orchestrator.plan("shop-2", check=False)

        assert await plans.get_by_filters() == []

    async def test__a_replan__keeps_the_kept_operations_where_they_were(
        self,
        orchestrator: PlanningOrchestrator[Shop, Schedule],
        problems: InMemoryRepository[Shop],
        plans: InMemoryRepository[Plan[Schedule]],
    ) -> None:
        """a2 grows longer: kept b1 stays put, every operation is placed, the plan is replaced."""
        # unkept, b1 is the operation this search moves to make room for a2
        first = await orchestrator.plan("shop-1", check=False)
        shop = await problems.get_by_id("shop-1")
        assert shop is not None
        longer = [
            op.model_copy(update={"duration": 6}) if op.ref == "a2" else op
            for op in shop.operations
        ]
        await problems.update_one(shop.model_copy(update={"operations": longer}))

        second = await orchestrator.replan("shop-1", ["b1"], check=False)

        assert self._placements(second)["b1"] == self._placements(first)["b1"]
        assert second.result.solution.unassigned == []
        assert second.result.solution.summary.score.hard == 0
        assert await plans.get_by_filters() == [second]

    async def test__a_replan_without_a_plan__raises_and_writes_nothing(
        self,
        orchestrator: PlanningOrchestrator[Shop, Schedule],
        plans: InMemoryRepository[Plan[Schedule]],
    ) -> None:
        """A re-plan needs the last plan of the problem; without one, nothing is solved."""
        with pytest.raises(RecordNotFoundError, match="shop-1"):
            await orchestrator.replan("shop-1", ["a1"], check=False)

        assert await plans.get_by_filters() == []
