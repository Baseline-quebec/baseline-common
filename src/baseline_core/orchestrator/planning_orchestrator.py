"""The planning flow: read a problem, solve it, and write its plan under the problem's id."""

from pydantic import BaseModel, ConfigDict

from baseline_core.core.repository.abstract_repository import (
    AbstractReadRepository,
    AbstractWriteRepository,
    RecordNotFoundError,
)
from baseline_core.service.optimization.optimization_service import (
    OptimizationResult,
    OptimizationService,
)


class Plan[Out: BaseModel](BaseModel):
    """A solve's result, stored under the id of the problem it solves."""

    model_config = ConfigDict(frozen=True)

    id: str
    """The id of the problem solved."""
    result: OptimizationResult[Out]
    """What the solve returned: its solution in keys is what a re-plan starts from."""


class PlanningOrchestrator[In: BaseModel, Out: BaseModel]:
    """Read a problem by its id, solve it, and write its plan under the same id."""

    def __init__(
        self,
        problems: AbstractReadRepository[In],
        service: OptimizationService[In, Out],
        plans: AbstractWriteRepository[Plan[Out]],
        previous_plans: AbstractReadRepository[Plan[Out]],
    ) -> None:
        """Keep the repositories and the service the flow chains.

        Args:
            problems: Reads the problems, each by its id.
            service: Solves a problem.
            plans: Writes the plans.
            previous_plans: Reads the plans written before, for a re-plan.
        """
        self.problems = problems
        self.service = service
        self.plans = plans
        self.previous_plans = previous_plans

    async def plan(self, problem_id: str, *, check: bool) -> Plan[Out]:
        """Read a problem, solve it, and write its plan, replacing any plan of that problem.

        Args:
            problem_id: The problem's id.
            check: Whether the pre-check runs before the search.

        Returns:
            The plan written.
        """
        problem = await self._read(self.problems, problem_id)
        result = self.service.solve(problem, check=check)

        return await self._write(problem_id, result)

    async def replan(self, problem_id: str, kept: list[str], *, check: bool) -> Plan[Out]:
        """Solve a problem again from its last plan, keeping some elements where they were.

        Args:
            problem_id: The problem's id, as it now is.
            kept: The keys of the elements that may not move, such as operations already started.
            check: Whether the pre-check runs before the search.

        Returns:
            The plan written, in place of the last one.
        """
        problem = await self._read(self.problems, problem_id)
        previous = await self._read(self.previous_plans, problem_id)
        assignments = previous.result.solution.assignments

        result = self.service.solve(
            problem,
            warm_start=[a for a in assignments if a.element not in kept],
            fixed=[a for a in assignments if a.element in kept],
            check=check,
        )

        return await self._write(problem_id, result)

    @staticmethod
    async def _read[T: BaseModel](repository: AbstractReadRepository[T], id_: str) -> T:
        """Read a record that the flow cannot go on without.

        Args:
            repository: The repository holding it.
            id_: Its id.

        Returns:
            The record.

        Raises:
            RecordNotFoundError: When the repository holds no record of that id.
        """
        record = await repository.get_by_id(id_)

        if record is None:
            msg = f"no {repository.content_type.__name__} {id_}"
            raise RecordNotFoundError(msg)

        return record

    async def _write(self, problem_id: str, result: OptimizationResult[Out]) -> Plan[Out]:
        """Write a result as the plan of its problem.

        Args:
            problem_id: The problem's id.
            result: What the solve returned.

        Returns:
            The plan written.
        """
        plan = Plan[Out](id=problem_id, result=result)
        await self.plans.upsert_one(plan)
        return plan
