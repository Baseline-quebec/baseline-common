"""A bench with planted rules: the instance, its model, the planted rules, a simulated planner."""

import random
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, Field, model_validator

from baseline_core.core.optimization.mining.candidate import (
    Candidate,
    CandidateFeature,
    ConstrainedDefinition,
)
from baseline_core.core.optimization.mining.plan import Plan, PlanView
from baseline_core.core.optimization.mining.sampler import KernelRestarts, KernelRestartsConfig
from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.declaration import (
    Assignment,
    IndexSet,
    IntegerDecision,
)
from baseline_core.core.optimization.problem_formulation.model import Model, Transition
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.search.kernel_solver import KernelSolverConfig


class Machine(BaseModel):
    """A machine."""

    code: str


class Task(BaseModel):
    """A weekly block of one operator's product."""

    name: str
    operator: str
    operators: list[str]
    """Operators whose no-overlap and lunch it counts in: two for a superposition."""
    product: str
    duration: int
    machines: list[Machine]
    workshop: str = ""
    """The operator's workshop: a grouping field the tables show."""
    special: bool = False
    blocked: list[int] = []
    """Absolute slots the block may not cover (hidden)."""
    spacing: int = 0
    """Least days between consecutive blocks of a lot it belongs to (0: none)."""
    spaced_lots: list[str] = []
    """The lots the spacing holds in: its operator's lot of its product, when spaced."""


class Problem(BaseModel):
    """The instance."""

    machines: list[Machine]
    tasks: list[Task]


class PlantedRule(BaseModel):
    """A bound the bench hides from the known model, on one operator, machine or product."""

    kind: Literal["operator", "machine", "product"]
    """What it bounds."""
    name: str
    """The operator, machine or product, as the instance names it."""
    feature: CandidateFeature
    """The bound, in the candidates' words."""
    bound: int
    """Slots after the day's first hour, a day index, or a count, as ``feature`` says."""


class BlockedHours(BaseModel):
    """Hours an operator or a workshop never works, on one day."""

    field: Literal["operator", "workshop"]
    """Which task field ``key`` reads."""
    key: str
    """The operator or the workshop."""
    day: int
    """The day index."""
    hours: list[int]
    """``[first, last)``: the hours blocked."""


class Spacing(BaseModel):
    """Least days between consecutive blocks of an operator's lot of a product."""

    product: str
    """The product."""
    days: int
    """The least days apart."""


class AllowedMachines(BaseModel):
    """The only machines a product may use."""

    product: str
    """The product."""
    machines: list[str]
    """The machine codes."""


class GenericRules(BaseModel):
    """Hidden rules of shapes the candidates have no word for: written on the data."""

    blocked: list[BlockedHours] = []
    """Blocked hours."""
    spacing: list[Spacing] = []
    """Day spacing."""
    allowed: list[AllowedMachines] = []
    """Allowed machines."""


class BenchInstanceConfig(BaseModel):
    """The size of the bench and its hidden rules."""

    PLANTED: ClassVar[list[PlantedRule]] = [
        PlantedRule(kind="operator", name="t2", feature=CandidateFeature.UNUSED_DAY, bound=4),
        PlantedRule(kind="operator", name="t5", feature=CandidateFeature.EARLIEST_START, bound=2),
        PlantedRule(kind="operator", name="t8", feature=CandidateFeature.MAX_PER_DAY, bound=2),
        PlantedRule(kind="operator", name="t11", feature=CandidateFeature.LATEST_END, bound=7),
        PlantedRule(kind="machine", name="C3", feature=CandidateFeature.UNUSED_DAY, bound=0),
        PlantedRule(kind="machine", name="L1", feature=CandidateFeature.LATEST_END, bound=6),
        PlantedRule(kind="machine", name="C5", feature=CandidateFeature.MAX_PER_DAY, bound=1),
        PlantedRule(kind="product", name="k7", feature=CandidateFeature.UNUSED_DAY, bound=3),
        PlantedRule(kind="product", name="k13", feature=CandidateFeature.EARLIEST_START, bound=2),
        PlantedRule(kind="product", name="k10", feature=CandidateFeature.LATEST_END, bound=8),
    ]
    """The default planted rules."""

    seed: int = 42
    """The seed of the durations."""
    operators: int = Field(default=14, gt=0)
    """Operators, 6 blocks each; operators 2i and 2i+1 share three products."""
    standard: int = Field(default=6, gt=0)
    """Standard machines (codes C)."""
    special: int = Field(default=2, gt=0)
    """Special machines (codes L)."""
    workshop_size: int = Field(default=4, gt=0)
    """Operators per workshop."""
    rules: list[PlantedRule] = PLANTED
    """The planted rules, posted on the true model as candidates."""
    generic: GenericRules = GenericRules()
    """The hidden rules written on the data."""

    @model_validator(mode="after")
    def _names_what_it_has(self) -> Self:
        """Refuse superpositions and rules naming an operator, machine or product the bench lacks.

        Returns:
            The configuration.

        Raises:
            ValueError: A superposition or a rule names something the bench does not have.
        """
        operators = {f"t{operator}" for operator in range(self.operators)}
        has = {
            "operator": operators,
            "machine": {f"C{i}" for i in range(self.standard)}
            | {f"L{i}" for i in range(self.special)},
            "product": {f"k{product}" for product in range((self.operators // 2) * 3)},
            "workshop": {f"d{t // self.workshop_size}" for t in range(self.operators)},
        }
        superposed = {task.split("b")[0] for pair in PlantedBench.SUPERPOSITIONS for task in pair}
        generic = self.generic
        named = (
            [["operator", operator] for operator in superposed]
            + [[rule.kind, rule.name] for rule in self.rules]
            + [[rule.field, rule.key] for rule in generic.blocked]
            + [["product", rule.product] for rule in generic.spacing]
            + [["product", rule.product] for rule in generic.allowed]
            + [["machine", machine] for rule in generic.allowed for machine in rule.machines]
        )
        missing = [f"{kind} {name}" for kind, name in named if name not in has[kind]]

        if missing:
            msg = f"the bench has no {missing}"
            raise ValueError(msg)

        return self


class BenchTables:
    """The bench in the planner's words: operators, products, lots, machines, days J1-J5."""

    ROLES: ClassVar[dict[str, str]] = {
        "operator": "Operateur",
        "product": "Produit",
        "machine": "Machine",
    }
    """Per kind of entity, its role in the tables."""
    DAY_LABELS: ClassVar[list[str]] = ["J1", "J2", "J3", "J4", "J5"]
    """The day labels."""

    @classmethod
    def name(cls, kind: str, entity: str) -> str:
        """Name an operator, a product or a machine as the tables write it.

        Args:
            kind: ``operator``, ``product`` or ``machine``.
            entity: Its name in the instance (``t5``, ``k7``, ``C3``).

        Returns:
            Its key (``op5``, ``P7``, ``M-C3``).
        """
        if kind == "machine":
            return "M-" + entity

        return {"operator": "op", "product": "P"}[kind] + entity[1:]

    @classmethod
    def view(cls, problem: Problem) -> PlanView:
        """Read an instance's elements as the tables do.

        Args:
            problem: The instance.

        Returns:
            The view: each task's operator, product, lot, section and workshop; each machine.
        """
        tasks = problem.tasks
        products = [cls.name("product", task.product) for task in tasks]
        lots = ["lot-" + task.operator[1:] for task in tasks]
        return PlanView(
            keys={
                "Operateur": [cls.name("operator", task.operator) for task in tasks],
                "Produit": products,
                "Lot": lots,
                "_section": [
                    f"{product}|{lot}" for product, lot in zip(products, lots, strict=True)
                ],
                "Atelier": ["A" + task.workshop[1:] for task in tasks],
            },
            target_role=cls.ROLES["machine"],
            target_names=[cls.name("machine", machine.code) for machine in problem.machines],
            period=PlantedBench.DAY,
            first_hour=PlantedBench.FIRST_HOUR,
        )

    @classmethod
    def rows(cls, problem: Problem, plans: list[Plan]) -> list[dict[str, str | int]]:
        """Write the plans as the planner's table: one row per placed task, absent ones left out.

        Args:
            problem: The instance the plans place.
            plans: The plans.

        Returns:
            The rows, a ``solution`` column numbering the plans.
        """
        view, day = cls.view(problem), PlantedBench.DAY
        roles = ["Operateur", "Produit", "Lot", "Atelier"]
        rows: list[dict[str, str | int]] = []

        for number, plan in enumerate(plans):
            out = set(plan.absent)

            for element, task in enumerate(problem.tasks):
                if element in out:
                    continue

                start = plan.starts[element]
                hour = PlantedBench.FIRST_HOUR + start % day
                rows.append({
                    "solution": number,
                    **{role: view.keys[role][element] for role in roles},
                    "Machine": view.target_names[plan.targets[element]],
                    "Jour": cls.DAY_LABELS[start // day],
                    "Debut": f"{hour:02d}:00",
                    "Fin": f"{hour + task.duration:02d}:00",
                })

        return rows


class BenchModelConfig(BaseModel):
    """Which data rules are posted, and the soft weights."""

    hidden_rules: bool = True
    """Post the blocked hours, day spacing and special turnaround the data holds."""
    turnaround: int = 0
    """Hours a special machine rests between two special tasks (``hidden_rules`` only; 0: none)."""
    weights: dict[str, int] = {"operator_machine_changes": 3, "lateness": 1}
    """The weight of each soft term."""


class BenchModelDefinition(AbstractModelDefinition[Problem]):
    """Within-day blocks, machines, no overlaps, lunch, two soft terms; the data's hidden rules."""

    schema = Problem

    def __init__(self, config: BenchModelConfig) -> None:
        """Keep the settings.

        Args:
            config: The rules to post and the weights.
        """
        self.config = config

    def define(self, model: Model[Problem]) -> None:
        """Post every rule.

        Args:
            model: The model to write into.
        """
        day_slots = PlantedBench.DAY
        tasks, machines = model.set(lambda s: s.tasks), model.set(lambda s: s.machines)
        duration = model.param(lambda task: task.duration, tasks)
        options = model.subsets(lambda task: task.machines, tasks, of=machines)
        start = model.var("start", tasks, span=duration, high=PlantedBench.DAYS * day_slots)
        machine = model.assign("machine", tasks, machines, among=options)
        teams = model.group_by(lambda task: task.operators, tasks)
        block = tasks.member
        day = start[block] // day_slots
        model.constraint((start.end(block) - 1) // day_slots == day, name="within_day")
        option = options[block].member
        model.constraint(model.count(option, machine[block] == option) > 0, name="machine_option")
        model.no_overlap(machine, start, name="no_machine_overlap")
        model.no_overlap(teams, start, name="no_operator_overlap")
        model.breaks(teams, start, name="lunch", period=day_slots, window=[3, 6], length=1)

        if self.config.hidden_rules:
            self._hidden(model, tasks, machine, start)

        weights = self.config.weights
        model.minimize(start[block] - day * day_slots, name="lateness", weight=weights["lateness"])

        for team in teams:
            order = model.sort(team, key=start)
            model.minimize(
                (start[order.previous] // day_slots == start[order.current] // day_slots)
                & (machine[order.previous] != machine[order.current]),
                name="operator_machine_changes",
                weight=weights["operator_machine_changes"],
            )

    def _hidden(
        self,
        model: Model[Problem],
        tasks: IndexSet[Task],
        machine: Assignment[Task],
        start: IntegerDecision,
    ) -> None:
        """Post the blocked hours, the day spacing within a lot, the special turnaround.

        Args:
            model: The model.
            tasks: The blocks.
            machine: Their machine.
            start: Their start.
        """
        block = tasks.member
        blocked = model.subsets(lambda task: task.blocked, tasks)
        slot = blocked[block].member
        covered = (start[block] <= slot) & (start.end(block) > slot)
        model.constraint(model.count(slot, covered) == 0, name="blocked_hours")
        spacing = model.param(lambda task: task.spacing, tasks)

        for lot in model.group_by(lambda task: task.spaced_lots, tasks):
            order = model.sort(lot, key=start)
            gap = (
                start[order.current] // PlantedBench.DAY
                - start[order.previous] // PlantedBench.DAY
            )
            model.constraint(gap >= spacing[order.current], name="day_spacing")

        if self.config.turnaround:
            hours = self.config.turnaround
            kind = Transition(
                kind=model.param(lambda task: task.product, tasks), same=hours, other=hours
            )
            model.no_overlap(
                machine,
                start,
                name="special_turnaround",
                where=model.param(lambda task: task.special, tasks),
                transition=kind,
            )


class PlantedBench:
    """One bench: its instances, its true model, a simulated planner's plans."""

    DAY: ClassVar[int] = 10
    """Slots in a day."""
    DAYS: ClassVar[int] = 5
    """Days in the week."""
    FIRST_HOUR: ClassVar[int] = 6
    """The hour of a day's first slot."""
    SUPERPOSITIONS: ClassVar[list[list[str]]] = [["t0b2", "t1b2"], ["t10b4", "t12b5"]]
    """``[leader, follower]``: one block two operators run, the follower merged into the leader."""
    RULE_CHOICES: ClassVar[dict[str, list[list[CandidateFeature | int | None]]]] = {
        "operator": [
            [CandidateFeature.UNUSED_DAY, None],
            [CandidateFeature.EARLIEST_START, 1],
            [CandidateFeature.EARLIEST_START, 2],
            [CandidateFeature.LATEST_END, 7],
            [CandidateFeature.LATEST_END, 8],
            [CandidateFeature.MAX_PER_DAY, 2],
        ],
        "standard": [
            [CandidateFeature.UNUSED_DAY, None],
            [CandidateFeature.LATEST_END, 7],
            [CandidateFeature.LATEST_END, 8],
            [CandidateFeature.MAX_PER_DAY, 1],
            [CandidateFeature.MAX_PER_DAY, 2],
        ],
        "special": [[CandidateFeature.UNUSED_DAY, None], [CandidateFeature.LATEST_END, 7]],
        "product": [
            [CandidateFeature.UNUSED_DAY, None],
            [CandidateFeature.EARLIEST_START, 2],
            [CandidateFeature.LATEST_END, 7],
            [CandidateFeature.LATEST_END, 8],
        ],
    }
    """Per kind of entity, the features a drawn rule may take and their bound (``None``: a day)."""

    def __init__(self, config: BenchInstanceConfig) -> None:
        """Keep the bench's size and hidden rules.

        Args:
            config: The configuration.
        """
        self.config = config

    @property
    def followers(self) -> list[str]:
        """The operators and products only a merged task carried: no drawn rule names them."""
        tasks = {task.name: task for task in self._tasks(self._machines())}
        names = []

        for leader, follower in self.SUPERPOSITIONS:
            names.append(tasks[follower].operator)

            if tasks[follower].product != tasks[leader].product:
                names.append(tasks[follower].product)

        return names

    def draw_rules(self, count: int, seed: int) -> list[PlantedRule]:
        """Draw planted rules on distinct operators, machines or products, feasible by design.

        Args:
            count: How many rules.
            seed: The seed of the draw.

        Returns:
            The rules.
        """
        config, rng, followers = self.config, random.Random(seed), set(self.followers)
        pool = (
            [["operator", f"t{t}"] for t in range(config.operators)]
            + [["standard", f"C{i}"] for i in range(config.standard)]
            + [["special", f"L{i}"] for i in range(config.special)]
            + [["product", f"k{k}"] for k in range((config.operators // 2) * 3)]
        )
        rules = []

        for kind, name in rng.sample(
            [entry for entry in pool if entry[1] not in followers], count
        ):
            feature, bound = rng.choice(self.RULE_CHOICES[kind])
            rules.append(
                PlantedRule.model_validate({
                    "kind": "machine" if kind in {"standard", "special"} else kind,
                    "name": name,
                    "feature": feature,
                    "bound": rng.randrange(self.DAYS) if bound is None else bound,
                })
            )

        return rules

    def draw_generic(self, counts: dict[str, int], seed: int) -> GenericRules:
        """Draw blocked hours, day spacing and allowed machines, on entities no planted rule names.

        Args:
            counts: How many of each kind, by kind (``operator_blocked``, ``spacing``, …).
            seed: The seed of the draw.

        Returns:
            The rules.
        """
        config, rng = self.config, random.Random(seed + 1)
        taken = {rule.name for rule in config.rules} | set(self.followers)
        operators = [f"t{t}" for t in range(config.operators) if f"t{t}" not in taken]
        products = [
            f"k{k}" for k in range((config.operators // 2) * 3) if f"k{k}" not in taken and k % 3
        ]
        workshops = sorted({f"d{t // config.workshop_size}" for t in range(config.operators)})
        blocked = []

        for operator in rng.sample(operators, counts.get("operator_blocked", 0)):
            first = rng.randrange(self.FIRST_HOUR + 1, self.FIRST_HOUR + self.DAY - 2)
            day = rng.randrange(self.DAYS)
            blocked.append(
                BlockedHours(field="operator", key=operator, day=day, hours=[first, first + 2])
            )

        for workshop in rng.sample(workshops, counts.get("workshop_blocked", 0)):
            first = rng.choice([self.FIRST_HOUR + 2, self.FIRST_HOUR + 7])
            day = rng.randrange(self.DAYS)
            blocked.append(
                BlockedHours(field="workshop", key=workshop, day=day, hours=[first, first + 1])
            )

        spaced = counts.get("spacing", 0)
        picked = rng.sample(products, spaced + counts.get("eligible", 0))
        standard = [f"C{i}" for i in range(config.standard)]
        return GenericRules(
            blocked=blocked,
            spacing=[Spacing(product=product, days=2) for product in picked[:spaced]],
            allowed=[
                AllowedMachines(product=product, machines=sorted(rng.sample(standard, 2)))
                for product in picked[spaced:]
            ],
        )

    def build(self) -> list[Problem]:
        """Build the analyst's instance, then the true one where each superposition is one task.

        Returns:
            ``[analyst, truth]``.
        """
        config = self.config
        machines = self._machines()
        tasks = self._tasks(machines)
        self._write_generic(machines, tasks)
        analyst = Problem(machines=machines, tasks=tasks)
        truth = analyst.model_copy(deep=True)
        restricted = {rule.product for rule in config.generic.allowed}

        for task in analyst.tasks:
            if task.product in restricted:
                task.machines = machines[: config.standard]

        self._merge_superpositions(truth)
        return [analyst, truth]

    def as_analyst(self, analyst: Problem, truth: Problem, plan: Plan) -> Plan:
        """Read a plan of the true instance in the analyst's tasks, each follower on its leader.

        Args:
            analyst: The analyst's instance, every task listed.
            truth: The true instance, each superposition one task.
            plan: A plan of ``truth``.

        Returns:
            The same plan, one entry per analyst task.
        """
        leader_of = {follower: leader for leader, follower in self.SUPERPOSITIONS}
        position = {task.name: k for k, task in enumerate(truth.tasks)}
        rows = [position[leader_of.get(task.name, task.name)] for task in analyst.tasks]
        out = set(plan.absent)
        return Plan(
            starts=[plan.starts[row] for row in rows],
            targets=[plan.targets[row] for row in rows],
            absent=[k for k, row in enumerate(rows) if row in out],
        )

    def true_model(
        self, truth: Problem, model: BenchModelConfig
    ) -> ConstrainedDefinition[Problem]:
        """Build the bench's model with its planted rules posted on it as candidates.

        Args:
            truth: The true instance, whose keys the rules name.
            model: The model's data rules and weights.

        Returns:
            The definition.
        """
        planted = BenchTruth(self.config).planted
        return ConstrainedDefinition(BenchModelDefinition(model), BenchTables.view(truth), planted)

    @staticmethod
    def planner_plans(bound: BoundModel, effort: int, n: int) -> list[Plan]:
        """Simulate a planner: the engine's search cut short; a plan leaving a block out dropped.

        Args:
            bound: The true model, bound to the true instance.
            effort: The solver's iteration cap, and its patience.
            n: How many plans to try.

        Returns:
            The plans that place every block.
        """
        solver = KernelSolverConfig(n_iters=effort, patience=effort, seed=500)
        plans = KernelRestarts(KernelRestartsConfig(samples=n, solver=solver)).sample(bound)
        return [plan for plan in plans if not plan.absent]

    def _machines(self) -> list[Machine]:
        """Build the standard machines, then the special ones.

        Returns:
            The machines.
        """
        config = self.config
        return [Machine(code=f"C{i}") for i in range(config.standard)] + [
            Machine(code=f"L{i}") for i in range(config.special)
        ]

    def _tasks(self, machines: list[Machine]) -> list[Task]:
        """Build six blocks per operator, the first a special one, durations drawn from the seed.

        Args:
            machines: The machines, standard first.

        Returns:
            The tasks.
        """
        config, rng = self.config, random.Random(self.config.seed)
        tasks = []

        for operator in range(config.operators):
            for number in range(6):
                special = number == 0
                tasks.append(
                    Task(
                        name=f"t{operator}b{number}",
                        operator=f"t{operator}",
                        operators=[f"t{operator}"],
                        product=f"k{(operator // 2) * 3 + number // 2}",
                        duration=3 if special else rng.choice([2, 2, 3]),
                        machines=machines[config.standard :]
                        if special
                        else machines[: config.standard],
                        workshop=f"d{operator // config.workshop_size}",
                        special=special,
                    )
                )

        by_name = {task.name: task for task in tasks}

        for leader, follower in self.SUPERPOSITIONS:
            by_name[follower].duration = by_name[leader].duration

        return tasks

    def _write_generic(self, machines: list[Machine], tasks: list[Task]) -> None:
        """Write the blocked hours, the spacing and the allowed machines on the tasks they cover.

        Args:
            machines: The machines.
            tasks: The tasks, edited in place.
        """
        generic = self.config.generic

        for rule in generic.blocked:
            first, last = rule.hours
            slots = [rule.day * self.DAY + hour - self.FIRST_HOUR for hour in range(first, last)]

            for task in tasks:
                if getattr(task, rule.field) == rule.key:
                    task.blocked = sorted({*task.blocked, *slots})

        spaced = {rule.product: rule.days for rule in generic.spacing}
        allowed = {rule.product: rule.machines for rule in generic.allowed}

        for task in tasks:
            if task.product in spaced:
                task.spacing, task.spaced_lots = (
                    spaced[task.product],
                    [f"{task.product}|{task.operator}"],
                )

            if task.product in allowed:
                task.machines = [
                    machine for machine in machines if machine.code in allowed[task.product]
                ]

    def _merge_superpositions(self, truth: Problem) -> None:
        """Merge each follower into its leader: both operators, blocked hours and spaced lots.

        Args:
            truth: The true instance, edited in place.
        """
        tasks = {task.name: task for task in truth.tasks}

        for leader, follower in self.SUPERPOSITIONS:
            merged, gone = tasks[leader], tasks[follower]
            merged.operators.append(gone.operator)
            merged.blocked = sorted({*merged.blocked, *gone.blocked})
            merged.spaced_lots = [*merged.spaced_lots, *gone.spaced_lots]
            merged.spacing = max(merged.spacing, gone.spacing)

        followers = {follower for _, follower in self.SUPERPOSITIONS}
        truth.tasks = [task for task in truth.tasks if task.name not in followers]


class PlantedGrade(BaseModel):
    """What became of one planted rule."""

    name: str
    """The rule, as a candidate names it."""
    found: Literal["rule", "question", "missed"]
    """Whether the extraction took it for a rule, asked about it, or missed it."""
    encoded: bool
    """Whether the constrain loop encoded it."""


class BenchTruth:
    """The planted rules in the tables' words, to post them and to grade candidates."""

    def __init__(self, config: BenchInstanceConfig) -> None:
        """Translate the planted rules once, the Wednesday closure last.

        Args:
            config: The bench, with its planted rules.
        """
        hourly = {CandidateFeature.EARLIEST_START, CandidateFeature.LATEST_END}
        self.planted = [
            Candidate(
                role=BenchTables.ROLES[rule.kind],
                key=BenchTables.name(rule.kind, rule.name),
                feature=rule.feature,
                value=rule.bound + PlantedBench.FIRST_HOUR
                if rule.feature in hourly
                else rule.bound,
                kept=1,
                plans=1,
            )
            for rule in config.rules
        ]
        self.planted.append(
            Candidate(
                role="*",
                key="*",
                feature=CandidateFeature.CLOSED_HOURS,
                value=2,
                hours=[PlantedBench.FIRST_HOUR + 4, PlantedBench.FIRST_HOUR + 5],
                kept=1,
                plans=1,
            )
        )

    def match(self, candidate: Candidate) -> int | None:
        """Find the planted rule a candidate states.

        Args:
            candidate: The candidate.

        Returns:
            The rule's position; ``None`` when it states none.
        """
        daily = {CandidateFeature.UNUSED_DAY, CandidateFeature.CLOSED_HOURS}

        for position, planted in enumerate(self.planted):
            same = [planted.role, planted.key, planted.feature]

            if same != [candidate.role, candidate.key, candidate.feature]:
                continue

            if planted.feature in daily and planted.value != candidate.value:
                continue

            return position

        return None

    def label(self, candidate: Candidate) -> str:
        """Label a candidate against the planted rules.

        Args:
            candidate: The candidate.

        Returns:
            ``vraie`` planted, ``effet`` a consequence of a planted rule, ``fausse`` neither.
        """
        if self.match(candidate) is not None:
            return "vraie"

        for planted in self.planted:
            same_key = [planted.role, planted.key] == [candidate.role, candidate.key]
            same_value = (
                planted.feature != CandidateFeature.UNUSED_DAY or planted.value == candidate.value
            )

            if same_key and planted.feature != candidate.feature:
                return "effet"

            if (
                candidate.role == "_section"
                and planted.feature == candidate.feature
                and same_value
            ):
                product, lot = candidate.key.split("|")

                if planted.key in {product, "op" + lot.split("-")[1]}:
                    return "effet"

        return "fausse"

    def grade(
        self, rules: list[Candidate], questions: list[Candidate], added: list[Candidate]
    ) -> list[PlantedGrade]:
        """Grade each planted rule: how the extraction reported it, whether the loop encoded it.

        Args:
            rules: The candidates the extraction took for rules.
            questions: The candidates it asked about.
            added: The candidates the constrain loop encoded.

        Returns:
            One grade per planted rule.
        """
        grades = []

        for position, planted in enumerate(self.planted):
            found: Literal["rule", "question", "missed"]

            if any(self.match(candidate) == position for candidate in rules):
                found = "rule"
            elif any(self.match(candidate) == position for candidate in questions):
                found = "question"
            else:
                found = "missed"

            encoded = any(self.match(candidate) == position for candidate in added)
            grades.append(PlantedGrade(name=planted.name, found=found, encoded=encoded))

        return grades
