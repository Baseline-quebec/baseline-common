"""A tiny shop the mining tests bind: jobs of two slots, two machines, two days of eight slots."""

from typing import ClassVar

from pydantic import BaseModel

from baseline_core.core.optimization.mining.plan import PlanView
from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.model import Model


class Machine(BaseModel):
    """A machine."""

    code: str


class Job(BaseModel):
    """A block of two slots, on one machine."""

    name: str
    operator: str
    duration: int = 2
    machines: list[Machine]


class Shop(BaseModel):
    """Jobs and the machines they may use."""

    machines: list[Machine]
    jobs: list[Job]

    @classmethod
    def of(cls, operators: list[str]) -> "Shop":
        """Build a shop of one job per operator, every job free to use both machines.

        Args:
            operators: Each job's operator.

        Returns:
            The shop.
        """
        machines = [Machine(code="A"), Machine(code="B")]
        jobs = [
            Job(name=f"j{i}", operator=op, machines=machines) for i, op in enumerate(operators)
        ]
        return cls(machines=machines, jobs=jobs)

    @property
    def view(self) -> PlanView:
        """The shop's plans read by operator and machine, a day of eight slots from hour 0."""
        return PlanView(
            keys={"op": [job.operator for job in self.jobs]},
            target_role="machine",
            target_names=[machine.code for machine in self.machines],
            period=ShopDefinition.DAY,
            first_hour=0,
        )


class ShopDefinition(AbstractModelDefinition[Shop]):
    """Within a day, no overlap on a machine; start early, prefer machine A."""

    schema = Shop
    DAY: ClassVar[int] = 8
    """Slots in a day; the horizon is two days."""

    def define(self, model: Model[Shop]) -> None:
        """Post the rules.

        Args:
            model: The model to write into.
        """
        jobs, machines = model.set(lambda s: s.jobs), model.set(lambda s: s.machines)
        day = self.DAY
        start = model.var(
            "start", jobs, span=model.param(lambda j: j.duration, jobs), high=2 * day, period=day
        )
        machine = model.assign(
            "machine", jobs, machines, among=model.subsets(lambda j: j.machines, jobs, of=machines)
        )
        job = jobs.member
        model.constraint((start.end(job) - 1) // day == start[job] // day, name="within_day")
        model.no_overlap(machine, start, name="no_machine_overlap")
        model.minimize(start[job], name="lateness", weight=1)
        model.minimize(machine[job], name="machine_use", weight=1)
