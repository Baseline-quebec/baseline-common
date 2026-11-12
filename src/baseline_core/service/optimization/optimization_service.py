"""The optimization service, for any model: bind, check, solve, map; it never names the problem."""

from pydantic import BaseModel, ConfigDict

from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.infeasibility.infeasibility_check import (
    InfeasibilityCheck,
)
from baseline_core.core.optimization.solver.infeasibility.precheck_issue import PrecheckIssue
from baseline_core.core.optimization.solver.model_solution import ModelSolution, VariableAssignment
from baseline_core.core.optimization.solver.output_mapper import OutputMapper
from baseline_core.core.optimization.solver.search.kernel_solver import KernelSolver


class OptimizationResult[Out: BaseModel](BaseModel):
    """What a solve returns: the solution in keys, the problem's output, the pre-check issues."""

    model_config = ConfigDict(frozen=True)

    solution: ModelSolution
    """The assignments, the elements left out, the summary: what a later solve takes back."""
    output: Out
    """The problem's output schema, filled from the solution."""
    precheck_issues: list[PrecheckIssue] | None
    """What the pre-check reported before the search; ``None`` when the solve ran without it."""


class OptimizationService[In: BaseModel, Out: BaseModel]:
    """Bind the problem once, pre-check it if asked, solve it, and fill the output schema."""

    def __init__(
        self,
        compiled: ModelCompiler[In],
        precheck: InfeasibilityCheck,
        solver: KernelSolver,
        mapper: OutputMapper[Out],
    ) -> None:
        """Keep everything a solve is configured with.

        Args:
            compiled: The model, compiled once.
            precheck: The pre-check, its levels and limits configured.
            solver: The solver.
            mapper: The problem's output schema, and how a solution fills it.
        """
        self.compiled = compiled
        self.precheck = precheck
        self.solver = solver
        self.mapper = mapper

    def solve(
        self,
        problem: In,
        warm_start: list[VariableAssignment] | None = None,
        fixed: list[VariableAssignment] | None = None,
        *,
        check: bool,
    ) -> OptimizationResult[Out]:
        """Bind the model to the problem, pre-check it if asked, solve, and map the output.

        Args:
            problem: The schema object the model reads.
            warm_start: Where the search begins, such as a previous solution's assignments.
            fixed: What the search may not change.
            check: Whether the pre-check runs on the bound model before the search.

        Returns:
            The solution in keys, the problem's output, and the pre-check issues.
        """
        bound = self.compiled.bind(problem)
        precheck_issues = self.precheck.run(bound) if check else None
        solution = self.solver.solve(bound, warm_start, fixed)

        return OptimizationResult(
            solution=solution,
            output=self.mapper.map(bound, solution),
            precheck_issues=precheck_issues,
        )
