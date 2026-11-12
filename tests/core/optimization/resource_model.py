"""A small model with every resource primitive switchable, and a brute-force judge of issues."""

import itertools

import numpy as np
from pydantic import BaseModel

from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.declaration import (
    IndexSet,
    IntegerDecision,
    Parameter,
)
from baseline_core.core.optimization.problem_formulation.model import Model, Transition
from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.infeasibility.precheck_issue import PrecheckIssue


class Machine(BaseModel):
    """A machine: its capacity, the operations it needs, a kind it refuses."""

    code: str
    cap: int = 1
    need: int = 0
    banned_kind: int = -1


class Op(BaseModel):
    """An operation: its team, length, machines, demand, kind, waits and whether it takes part."""

    ref: str
    team: str = "X"
    duration: int = 1
    machines: list[str]
    demand: int = 1
    kind: int = 0
    same: int = 0
    other: int = 0
    takes: int = 1


class Shop(BaseModel):
    """The machines and the operations."""

    machines: list[Machine]
    operations: list[Op]


class TeamBreak(BaseModel):
    """The break each team keeps every period it works, posted as ``brk``."""

    period: int
    window: list[int]
    length: int


class ResourceModelDefinition(AbstractModelDefinition[Shop]):
    """The rules named in ``rules``, over ``Shop``; ``high`` ends the horizon, if timed."""

    schema = Shop

    def __init__(self, rules: frozenset[str], high: int, breaks: TeamBreak | None = None) -> None:
        """Keep the rules to post, the horizon and the break each team keeps, if any.

        Args:
            rules: The rule names to post, among those ``define`` knows; ``high`` 0: no time.
            high: Where the horizon ends; 0: no time.
            breaks: The break of the teams of ``team_overlap``, if any.
        """
        self.rules, self.high, self.breaks = rules, high, breaks

    def define(self, model: Model[Shop]) -> None:
        """Post the rules switched on.

        Args:
            model: The model to post on.
        """
        on = self.rules
        ops = model.set(lambda s: s.operations, key="ref")
        machines = model.set(lambda s: s.machines, key="code")

        kind = model.param(lambda op: op.kind, ops)
        machine = model.assign(
            "machine", ops, machines, among=model.subsets(lambda op: op.machines, ops, of=machines)
        )
        demand = model.param(lambda op: op.demand, ops)
        capacity = model.param(lambda each: each.cap, machines)

        if "ban" in on:
            for group in machine.groups:
                model.constraint(
                    kind[group.member]
                    != model.param(lambda each: each.banned_kind, machines)[group.target],
                    name="ban",
                )

        if "machine_min" in on:
            for group in machine.groups:
                need = model.param(lambda each: each.need, machines)
                model.constraint(group.size >= need[group.target], name="machine_min")

        if "bin" in on:
            model.bin_packing(machine, demand=demand, capacity=capacity, name="bin")

        if not self.high:
            return

        start = model.var(
            "start", ops, span=model.param(lambda op: op.duration, ops), high=self.high
        )

        takes = model.param(lambda op: op.takes, ops) if "where" in on else None
        gap = Transition(
            kind, model.param(lambda op: op.same, ops), model.param(lambda op: op.other, ops)
        )

        if "machine_overlap" in on:
            model.no_overlap(
                machine,
                start,
                name="machine_overlap",
                where=takes,
                transition=gap if "transition" in on else None,
            )

        if "team_overlap" in on:
            self._post_team_overlap(model, ops, start, takes)

        if "cumulative" in on:
            model.cumulative(machine, start, demand=demand, capacity=capacity, name="cumulative")

    def _post_team_overlap(
        self,
        model: Model[Shop],
        ops: IndexSet[Op],
        start: IntegerDecision,
        takes: Parameter[Op] | None,
    ) -> None:
        """No two operations of a team at once; each team keeps its break, if any.

        Args:
            model: The model to post on.
            ops: The operations.
            start: Their start.
            takes: Per operation, whether it takes part; ``None``: every one.
        """
        teams = model.group_by(lambda op: op.team, ops)
        model.no_overlap(teams, start, name="team_overlap", where=takes)

        if self.breaks is not None:
            model.breaks(
                teams,
                start,
                name="brk",
                period=self.breaks.period,
                window=self.breaks.window,
                length=self.breaks.length,
                where=takes,
            )


class Judge:
    """Every placement of a shop, counted per constraint, to judge proofs by brute force."""

    def __init__(self, compiled: ModelCompiler[Shop], high: int) -> None:
        """Keep the compiled model and the horizon.

        Args:
            compiled: The compiled resource model.
            high: Where the horizon ends; 0: no time.
        """
        self.compiled, self.high = compiled, high

    def counts(self, shop: Shop) -> np.ndarray:
        """Per placement of the whole shop, the raw count of each constraint.

        Args:
            shop: The shop.

        Returns:
            One row per placement.
        """
        kernel = self.compiled.bind(shop).kernel
        code = {machine.code: position for position, machine in enumerate(shop.machines)}
        domains = [
            [
                (start, code[name])
                for start in (range(self.high - op.duration + 1) if self.high else [0])
                for name in op.machines
            ]
            for op in shop.operations
        ]

        rows = []

        for combination in itertools.product(*domains):
            arrays = {"machine": np.array([target for _, target in combination], dtype=np.int64)}

            if self.high:
                arrays["start"] = np.array([start for start, _ in combination], dtype=np.int64)

            kernel.load(**arrays)
            rows.append(kernel.full_counts)

        return np.array(rows, dtype=np.int64).reshape(-1, len(self.compiled.model.names))

    def wrong(self, shop: Shop, issues: list[PrecheckIssue]) -> list[PrecheckIssue]:
        """The proven issues whose named constraints a placement satisfies all at once.

        Args:
            shop: The shop.
            issues: The pre-check's issues.

        Returns:
            The false proofs and the explanations that do not suffice.
        """
        table = self.counts(shop)
        names = self.compiled.model.names
        return [
            issue
            for issue in issues
            if issue.proven
            and (
                table[:, [names.index(name) for name in issue.constraints]].sum(axis=1) == 0
            ).any()
        ]
