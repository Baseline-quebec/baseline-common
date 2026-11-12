import pytest

from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.infeasibility.infeasibility_check import (
    InfeasibilityCheck,
    PrecheckConfig,
)
from baseline_core.core.optimization.solver.infeasibility.precheck_issue import IssueKind
from baseline_core.core.optimization.solver.model_solution import VariableAssignment
from baseline_core.core.optimization.solver.output_mapper import OutputMapper
from baseline_core.core.optimization.solver.search.kernel_solver import (
    KernelSolver,
    KernelSolverConfig,
)
from baseline_core.service.optimization.optimization_service import (
    OptimizationResult,
    OptimizationService,
)
from tests.core.optimization.job_shop_model import JobShopModelDefinition, Schedule, Shop


@pytest.fixture(scope="module")
def service() -> OptimizationService[Shop, Schedule]:
    """The job shop's service: every pre-check level under a budget, a short search.

    Returns:
        The service.
    """
    compiled = ModelCompiler(JobShopModelDefinition(), cythonize=False)
    precheck = InfeasibilityCheck(PrecheckConfig(time_budget_seconds=20))
    solver = KernelSolver(KernelSolverConfig(n_iters=200, patience=100))
    return OptimizationService(compiled, precheck, solver, OutputMapper(Schedule))


@pytest.fixture
def late() -> Shop:
    """Two jobs, one operation of which cannot end by its due time: 4 h due at 3.

    Returns:
        The shop.
    """
    return Shop.of([
        ["a1", "A", 1, 3, ["M1"], 0, 20],
        ["a2", "A", 2, 2, ["M2"], 0, 20],
        ["b1", "B", 1, 4, ["M1", "M2"], 0, 20],
        ["b2", "B", 2, 4, ["M1"], 0, 3],
    ])


@pytest.fixture
def feasible() -> Shop:
    """Two jobs that fit on their machines.

    Returns:
        The shop.
    """
    return Shop.of([
        ["a1", "A", 1, 4, ["M1"], 0, 20],
        ["a2", "A", 2, 3, ["M2"], 0, 20],
        ["b1", "B", 1, 5, ["M2"], 0, 20],
        ["b2", "B", 2, 2, ["M1"], 0, 20],
    ])


class TestOptimizationService:
    @staticmethod
    def _rows(result: OptimizationResult[Schedule]) -> dict[str, list[int | str]]:
        """Each scheduled operation's start, end and machine, by its ref.

        Args:
            result: What the service returned.

        Returns:
            ``[start, end, machine]`` per operation.
        """
        return {op.ref: [op.start, op.end, op.machine] for op in result.output.operations}

    def test__a_checked_solve__returns_what_the_pre_check_proved(
        self, service: OptimizationService[Shop, Schedule], late: Shop
    ) -> None:
        """Checked, the result carries the operation proven late, beside the solution."""
        result = service.solve(late, check=True)
        proven = [
            (issue.kind, issue.entity, issue.constraints)
            for issue in result.precheck_issues or []
            if issue.proven
        ]

        assert proven == [(IssueKind.NO_PLACEMENT, "b2", ["time_window"])]
        assert len(result.solution.assignments) + len(result.solution.unassigned) == 4
        assert "b2" in result.solution.unassigned

    def test__an_unchecked_solve__says_no_pre_check_ran(
        self, service: OptimizationService[Shop, Schedule], late: Shop
    ) -> None:
        """Unchecked, the pre-check issues are ``None``: not run, rather than run and clean."""
        assert service.solve(late, check=False).precheck_issues is None

    def test__the_output__is_the_solution_mapped_on_the_problem(
        self, service: OptimizationService[Shop, Schedule], feasible: Shop
    ) -> None:
        """Each operation of the output is an assignment of the solution, its end added."""
        result = service.solve(feasible, check=True)
        durations = {op.ref: op.duration for op in feasible.operations}
        expected = {
            a.element: [a.value, (a.value or 0) + durations[a.element], a.target]
            for a in result.solution.assignments
        }

        assert result.precheck_issues is not None
        assert [issue for issue in result.precheck_issues if issue.proven] == []
        assert result.solution.summary.score.hard == 0
        assert result.solution.unassigned == []
        assert self._rows(result) == expected

    def test__a_fixed_operation__ends_where_it_was_fixed(
        self, service: OptimizationService[Shop, Schedule], feasible: Shop
    ) -> None:
        """An operation fixed on a machine at a time is there in the output, nothing broken."""
        fixed = [VariableAssignment(element="b2", value=12, target="M1")]
        result = service.solve(feasible, fixed=fixed, check=False)

        assert self._rows(result)["b2"] == [12, 14, "M1"]
        assert result.solution.summary.score.hard == 0

    def test__a_warm_start__gives_a_result_no_worse_than_it(
        self, service: OptimizationService[Shop, Schedule], feasible: Shop
    ) -> None:
        """A solve warm-started from a previous solution's assignments scores no worse."""
        first = service.solve(feasible, check=False)
        second = service.solve(feasible, warm_start=first.solution.assignments, check=False)

        assert second.solution.summary.score <= first.solution.summary.score
