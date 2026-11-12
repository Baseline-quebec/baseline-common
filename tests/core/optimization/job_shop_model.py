"""A job shop over its own schema: operations, machines, the order inside a job."""

from typing import ClassVar

from pydantic import AliasPath, BaseModel, ConfigDict, Field

from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.declaration import (
    IndexSet,
    IntegerDecision,
    Parameter,
)
from baseline_core.core.optimization.problem_formulation.model import Model


class Machine(BaseModel):
    """A machine, named by its code."""

    code: str


class Operation(BaseModel):
    """One step of a job: its order in the job, how long it takes, on which machines, and when."""

    ref: str
    job: str
    step: int
    duration: int
    machines: list[str]
    release: int
    due: int


class Shop(BaseModel):
    """The job shop."""

    machines: list[Machine]
    operations: list[Operation]

    @classmethod
    def of(cls, operations: list[list[object]]) -> "Shop":
        """A shop on M1 and M2, one ``[ref, job, step, duration, machines, release, due]``.

        Args:
            operations: One row per operation.

        Returns:
            The shop.
        """
        keys = ["ref", "job", "step", "duration", "machines", "release", "due"]
        return cls(
            machines=[Machine(code="M1"), Machine(code="M2")],
            operations=[
                Operation.model_validate(dict(zip(keys, row, strict=True))) for row in operations
            ],
        )


class ScheduledOperation(BaseModel):
    """One operation, where the solution put it: read off one ``assigned`` entry."""

    # the pydantic mypy plugin types an AliasPath field only when its name validates too
    model_config = ConfigDict(validate_by_name=True)

    ref: str = Field(validation_alias=AliasPath("element", "ref"))
    start: int = Field(validation_alias="value")
    end: int
    machine: str = Field(validation_alias=AliasPath("target", "code"))


class Schedule(BaseModel):
    """The job shop's output: every operation placed."""

    operations: list[ScheduledOperation] = Field(validation_alias="assigned")


class JobShopModelDefinition(AbstractModelDefinition[Shop]):
    """Four hard rules over ``Shop``."""

    schema = Shop

    def define(self, model: Model[Shop]) -> None:
        """Post every rule.

        Args:
            model: The model to post on.
        """
        operations = model.set(lambda s: s.operations, key="ref")
        machines = model.set(lambda s: s.machines, key="code")

        duration = model.param(lambda op: op.duration, operations)
        step = model.param(lambda op: op.step, operations)
        release = model.param(lambda op: op.release, operations)
        due = model.param(lambda op: op.due, operations)
        options = model.subsets(lambda op: op.machines, operations, of=machines)

        start = model.var("start", operations, span=duration, high=40)
        machine = model.assign("machine", operations, machines)

        operation = operations.member
        inside = (release[operation] <= start[operation]) & (
            start.end(operation) <= due[operation]
        )
        model.constraint(inside, name="time_window")

        option = options[operation].member
        model.constraint(
            model.count(option, machine[operation] == option) > 0, name="machine_option"
        )

        model.no_overlap(machine, start, name="no_machine_overlap")
        self._post_job_order(model, operations, start, step)

    @staticmethod
    def _post_job_order(
        model: Model[Shop],
        operations: IndexSet[Operation],
        start: IntegerDecision,
        step: Parameter[Operation],
    ) -> None:
        """The operations of a job run one after another, in the order of their steps.

        Args:
            model: The model.
            operations: The operations.
            start: Their start.
            step: Per operation, its order in its job.
        """
        for job in model.group_by(lambda op: op.job, operations):
            operation, other = job.pair
            operation_first = (step[operation] >= step[other]) | (
                start.end(operation) <= start[other]
            )
            other_first = (step[other] >= step[operation]) | (start.end(other) <= start[operation])
            model.constraint(operation_first & other_first, name="job_order")


class EarlyStartModelDefinition(AbstractModelDefinition[Shop]):
    """Operations on their machines, never overlapping, and an hour costing more than a rule."""

    schema = Shop

    HOUR_PRICE: ClassVar[int] = 1_000_000
    """The soft cost of starting one hour later, above any fixed hard-to-soft exchange rate."""

    def define(self, model: Model[Shop]) -> None:
        """Post the machines, the overlap rule and the price of a late start.

        Args:
            model: The model to post on.
        """
        operations = model.set(lambda s: s.operations, key="ref")
        machines = model.set(lambda s: s.machines, key="code")

        duration = model.param(lambda op: op.duration, operations)
        options = model.subsets(lambda op: op.machines, operations, of=machines)

        start = model.var("start", operations, span=duration, high=40)
        machine = model.assign("machine", operations, machines, among=options)

        model.no_overlap(machine, start, name="no_machine_overlap")
        model.minimize(start[operations.member], name="early", weight=self.HOUR_PRICE)
