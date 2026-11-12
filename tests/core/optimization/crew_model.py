"""A workshop run by crews, over tasks, stations and routes: a test model of the engine."""

from collections.abc import Iterator
from enum import StrEnum
from functools import cached_property
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, RootModel

from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.declaration import (
    Assignment,
    IndexSet,
    IntegerDecision,
    Member,
    Parameter,
    SubsetFamily,
)
from baseline_core.core.optimization.problem_formulation.expression import Expr
from baseline_core.core.optimization.problem_formulation.model import Model, Tie, Transition


class TaskKind(StrEnum):
    """What a task needs from its station.

    Attributes:
        STANDARD: Any station of its route.
        CLEAN: A clean station, cleaned between two clean tasks.
    """

    STANDARD = "S"
    CLEAN = "C"


class Spacing(StrEnum):
    """How the runs of one order are spaced over the days.

    Attributes:
        NONE: No relation.
        SAME_DAY: Every run on one day, a gap apart.
        CONSECUTIVE_DAYS: One run per day, on back-to-back days.
        DAY_APART: At least one free day between two runs.
    """

    NONE = "none"
    SAME_DAY = "same_day"
    CONSECUTIVE_DAYS = "consecutive_days"
    DAY_APART = "day_apart"


class Window(BaseModel):
    """Slots a task may take, ``[start, end)``."""

    start: int
    end: int


class SoftPeriod(BaseModel):
    """Slots a crew would rather keep free, ``[start, end)``."""

    model_config = ConfigDict(frozen=True)

    crew_id: str
    start: int
    end: int


class Reservation(BaseModel):
    """A start, and maybe a station, held for a task."""

    model_config = ConfigDict(frozen=True)

    SLOTS_PER_DAY: ClassVar[int] = 24
    """Slots in a day."""

    day_index: int
    start_slot: int
    station_id: str | None = None

    @property
    def absolute_start(self) -> int:
        """The start on the weekly grid."""
        return self.day_index * self.SLOTS_PER_DAY + self.start_slot


class Station(BaseModel):
    """A station, holding ``capacity`` tasks at once."""

    id_: str
    name: str
    capacity: int = 1


class Duration(RootModel[int]):
    """A number of slots."""


class Step(BaseModel):
    """A task done at one station, for a duration."""

    id_: str
    name: str
    station_id: str
    duration: Duration


class Alternatives(BaseModel):
    """Steps of which a task takes one."""

    options: list[Step]


class Route(BaseModel):
    """How a task is done: its first step, or alternatives for it."""

    id_: str
    name: str
    steps: list[Step | Alternatives]

    @cached_property
    def options(self) -> list[Step]:
        """The steps the task may take first, one per station."""
        first = self.steps[0]
        return first.options if isinstance(first, Alternatives) else [first]


class Task(BaseModel):
    """One task: its route, its windows, its crew, its order and its run in the order."""

    SPLIT_MARKER: ClassVar[str] = "_part"
    """Separates a split task's base id from its part number."""

    id_: str
    name: str
    route_id: str
    windows: list[Window] = Field(min_length=1)
    product: str
    crew_id: str
    kind: TaskKind
    order_key: str
    run: int = Field(default=1, ge=1)
    spacing: Spacing = Spacing.NONE
    clean_gap_same_product: int = Field(default=1, ge=0)
    clean_gap_other_product: int = Field(default=2, ge=0)
    helper_crews: list[str] = Field(default_factory=list)
    companion_of: str | None = None
    preferred_stations: list[list[str]] = Field(default_factory=list)
    crew_soft_periods: list[SoftPeriod] = Field(default_factory=list)
    reservation: Reservation | None = None

    @property
    def is_clean(self) -> bool:
        """Whether it needs a clean station."""
        return self.kind == TaskKind.CLEAN

    @property
    def crew_ids(self) -> list[str]:
        """The crews it keeps busy, helpers included; none for a companion task."""
        return [] if self.companion_of is not None else [self.crew_id, *self.helper_crews]

    @property
    def split_keys(self) -> list[str]:
        """The split task this part belongs to, by its base id; none for a whole task."""
        marker_at = self.id_.rfind(self.SPLIT_MARKER)
        part = self.id_[marker_at + len(self.SPLIT_MARKER) :]
        return [self.id_[:marker_at]] if marker_at != -1 and part.isdigit() else []

    @property
    def spacing_keys(self) -> list[str]:
        """The order whose day spacing it keeps; none for a companion task."""
        return [] if self.companion_of is not None else [self.order_key]


class Tasks(RootModel[list[Task]]):
    """The tasks: a root list, read by the engine as the list it wraps."""

    # BaseModel iterates its field pairs; a root list iterates its items, as pydantic shows
    def __iter__(self) -> Iterator[Task]:  # type: ignore[override]
        """Each task, in order."""
        return iter(self.root)


class Workshop(BaseModel):
    """The workshop: tasks, stations, routes, and the pairs of tasks that start together."""

    tasks: Tasks
    stations: list[Station]
    routes: list[Route]
    co_start_groups: list[list[str]] = Field(default_factory=list)


class CrewModelConfig(BaseModel):
    """Settings of the workshop rules."""

    slots_per_day: int = 24
    """Slots in a day; one slot is one hour."""
    n_days: int = 5
    """Days in the week."""
    break_hours: list[int] = [11, 12, 13]
    """Hours of the day among which a crew's break is taken."""
    min_break_hours: int = 1
    """Consecutive free hours a crew needs among the break hours."""
    split_min_gap_hours: int = 1
    """Least gap between the parts of a split task."""
    split_same_day: bool = True
    """Parts of a split task share a day (else they take different days)."""
    same_day_min_gap_hours: int = 1
    """Least gap between runs of an order spaced on the same day."""
    min_days_apart: int = 2
    """Fewest days between runs of an order spaced a day apart."""
    weights: dict[str, int] = {
        "station_preference": 50,
        "crew_soft_period": 10,
        "crew_station_changes": 20,
        "station_utilization": 20,
        "clean_product_chaining": 100,
    }
    """Weight of each soft rule."""


class CrewModelDefinition(AbstractModelDefinition[Workshop]):
    """Hard and soft rules of a workshop run by crews."""

    schema = Workshop
    NOT_HELD: ClassVar[int] = 1 << 30
    """Rank of a preferred pool the task's station is not in."""
    NO_NEXT_DAY: ClassVar[int] = 1 << 30
    """Stands for "no next run" in the earliest day of the later runs."""

    def __init__(self, config: CrewModelConfig) -> None:
        """Keep the settings the rules read.

        Args:
            config: The settings.
        """
        self.config = config

    def define(self, model: Model[Workshop]) -> None:
        """Declare the sets, parameters and decisions, then post every rule.

        Args:
            model: The model to post on.
        """
        config = self.config
        tasks = model.set(lambda s: s.tasks, key="id_")
        stations = model.set(lambda s: s.stations, key="id_")
        routes = model.set(lambda s: s.routes, key="id_")

        duration = model.param(lambda task: routes[task.route_id].options[0].duration, tasks)
        start = model.var(
            "start",
            tasks,
            span=duration,
            high=config.n_days * config.slots_per_day,
            period=config.slots_per_day,
        )

        options = model.subsets(
            lambda task: [option.station_id for option in routes[task.route_id].options],
            tasks,
            of=stations,
        )
        station = model.assign("station", tasks, stations, among=options)
        crews = model.group_by(lambda task: task.crew_ids, tasks)

        self.post_window_coverage(model, tasks, start)
        self._post_station_option(model, tasks, options, station)
        self._post_overlaps_and_break(model, station, crews, start)
        self._post_split_task_spacing(model, tasks, start)
        self._post_run_step(model, tasks, start)
        self._post_same_day_gap(model, tasks, start)
        self._post_co_start(model, tasks, start)
        self._post_clean_rules(model, tasks, station, start)
        self._post_crew_soft_period(model, crews, start)
        self._post_crew_station_changes(model, crews, start, station)
        self._post_station_utilization(model, station)
        self._post_station_preference(model, tasks, stations, station)

    def _day(self, start: IntegerDecision, task: Expr) -> Expr:
        """The day a task starts on.

        Args:
            start: The start decision.
            task: A task.

        Returns:
            Its start divided by the slots of a day.
        """
        return start[task] // self.config.slots_per_day

    @staticmethod
    def _gap(model: Model[Workshop], start: IntegerDecision, task: Expr, other: Expr) -> Expr:
        """The free time between two tasks, ``0`` when they overlap.

        Args:
            model: The model.
            start: The start decision.
            task: A task.
            other: Another task.

        Returns:
            How long one waits after the other ends.
        """
        return model.largest(0, start[other] - start.end(task), start[task] - start.end(other))

    @staticmethod
    def post_window_coverage(
        model: Model[Workshop], tasks: IndexSet[Task], start: IntegerDecision
    ) -> None:
        """Each task lies whole inside one of its windows.

        Args:
            model: The model.
            tasks: The tasks.
            start: Their start.
        """
        task = tasks.member
        window = model.within(lambda each: each.windows, task)
        inside = (window.start <= start[task]) & (start.end(task) <= window.end)
        model.constraint(model.count(window, inside) > 0, name="window_coverage")

    @staticmethod
    def _post_station_option(
        model: Model[Workshop],
        tasks: IndexSet[Task],
        options: SubsetFamily[Any],
        station: Assignment[Task],
    ) -> None:
        """Each task takes one of its route's stations.

        Args:
            model: The model.
            tasks: The tasks.
            options: Per task, the stations its route may take.
            station: Per task, its station.
        """
        task = tasks.member
        option = options[task].member
        model.constraint(model.count(option, station[task] == option) > 0, name="station_option")

    def _post_overlaps_and_break(
        self,
        model: Model[Workshop],
        station: Assignment[Task],
        crews: SubsetFamily[Task],
        start: IntegerDecision,
    ) -> None:
        """No station or crew holds two tasks at once; a crew keeps a free break hour daily.

        Args:
            model: The model.
            station: Per task, its station.
            crews: Per crew, its tasks.
            start: The start decision.
        """
        config = self.config
        model.no_overlap(station, start, name="no_station_overlap")

        model.no_overlap(crews, start, name="no_crew_overlap")
        model.tie("no_crew_overlap", Tie.RELEASED)

        model.breaks(
            crews,
            start,
            name="crew_break",
            period=config.slots_per_day,
            window=[config.break_hours[0], config.break_hours[-1] + 1],
            length=max(config.min_break_hours, 1),
        )

    def _post_split_task_spacing(
        self, model: Model[Workshop], tasks: IndexSet[Task], start: IntegerDecision
    ) -> None:
        """The parts of a split task share a day, a gap apart (or take different days).

        Args:
            model: The model.
            tasks: The tasks.
            start: The start decision.
        """
        config = self.config

        for split in model.group_by(lambda task: task.split_keys, tasks):
            part, other = split.pair
            same_day = self._day(start, part) == self._day(start, other)
            together = same_day & (
                self._gap(model, start, part, other) >= config.split_min_gap_hours
            )
            apart = self._day(start, part) != self._day(start, other)

            model.constraint(
                together if config.split_same_day else apart, name="split_task_spacing"
            )

        model.tie("split_task_spacing", Tie.GROUPED)

    def _post_run_step(
        self, model: Model[Workshop], tasks: IndexSet[Task], start: IntegerDecision
    ) -> None:
        """A first task of its run is followed by the next run neither too soon nor too late.

        Args:
            model: The model.
            tasks: The tasks.
            start: The start decision.
        """
        run = model.param(lambda task: task.run, tasks)
        spacing = model.param(lambda task: task.spacing, tasks)

        days_to_next = {
            Spacing.SAME_DAY: [0, 0],
            Spacing.CONSECUTIVE_DAYS: [1, 1],
            Spacing.DAY_APART: [self.config.min_days_apart, self.config.n_days],
        }
        bounds = [days_to_next.get(kind, [0, self.config.n_days]) for kind in Spacing]
        fewest = model.lookup_table([lowest for lowest, _ in bounds])
        most = model.lookup_table([highest for _, highest in bounds])

        for order in model.group_by(lambda task: task.spacing_keys, tasks):
            current = order.member
            next_day = self._next_day(model, start, run, current)
            step = next_day - self._day(start, current)

            lowest, highest = fewest[spacing[current]], most[spacing[current]]
            tight = (lowest > 0) | (highest < self.config.n_days - 1)
            out_of_bounds = (step < lowest) | (step > highest)

            first = self._is_first(model, run, current, order.member)
            bad_step = tight & first & (next_day < self.NO_NEXT_DAY) & out_of_bounds
            model.constraint(~bad_step, name="run_day_spacing")

    def _post_same_day_gap(
        self, model: Model[Workshop], tasks: IndexSet[Task], start: IntegerDecision
    ) -> None:
        """Two tasks of an order from different runs on one day keep the gap the spacing asks.

        Args:
            model: The model.
            tasks: The tasks.
            start: The start decision.
        """
        config = self.config
        run = model.param(lambda task: task.run, tasks)
        spacing = model.param(lambda task: task.spacing, tasks)

        min_gap = model.lookup_table([
            config.same_day_min_gap_hours if kind is Spacing.SAME_DAY else 0 for kind in Spacing
        ])

        for order in model.group_by(lambda task: task.spacing_keys, tasks):
            task, other = order.member, order.member
            gap = min_gap[spacing[task]]
            close = (
                (gap > 0)
                & (other > task)
                & (run[task] != run[other])
                & (self._day(start, task) == self._day(start, other))
                & (self._gap(model, start, task, other) < gap)
            )
            model.constraint(~close, name="run_day_spacing")

    def _next_day(
        self,
        model: Model[Workshop],
        start: IntegerDecision,
        run: Parameter[Task],
        current: Member[Task],
    ) -> Expr:
        """The day the run after ``current`` starts on (``NO_NEXT_DAY`` when none follows).

        Args:
            model: The model.
            start: The start decision.
            run: Per task, its run in its order.
            current: A free member of an order: the task judged.

        Returns:
            The earliest day of a first task of a later run.
        """
        order = current.collection
        candidate = order.member

        day, current_day = self._day(start, candidate), self._day(start, current)
        later = (day > current_day) | ((day == current_day) & (run[candidate] > run[current]))
        starts_next_run = (
            self._is_first(model, run, candidate, order.member)
            & (run[candidate] != run[current])
            & later
        )

        return model.min(candidate, model.if_then_else(starts_next_run, day, self.NO_NEXT_DAY))

    @staticmethod
    def _is_first(
        model: Model[Workshop], run: Parameter[Task], task: Member[Any], rival: Member[Any]
    ) -> Expr:
        """Whether ``task`` is the first of its run, ``rival`` ranging over the order.

        Args:
            model: The model.
            run: Per task, its run in its order.
            task: A task.
            rival: A free member of the same order.

        Returns:
            No task of lower position shares its run.
        """
        return model.count(rival, (run[rival] == run[task]) & (rival < task)) == 0

    @staticmethod
    def _post_co_start(
        model: Model[Workshop], tasks: IndexSet[Task], start: IntegerDecision
    ) -> None:
        """The tasks of a co-start group start together.

        Args:
            model: The model.
            tasks: The tasks.
            start: The start decision.
        """
        pairs = model.set(lambda s: s.co_start_groups)

        for group in model.subsets(lambda pair: pair, pairs, of=tasks):
            task, other = group.pair
            model.constraint(start[task] == start[other], name="co_start")

        model.tie("co_start", Tie.PAIRED)

    def _post_clean_rules(
        self,
        model: Model[Workshop],
        tasks: IndexSet[Task],
        station: Assignment[Task],
        start: IntegerDecision,
    ) -> None:
        """Two clean tasks at one station leave the cleaning their products ask; cleanings are few.

        Args:
            model: The model.
            tasks: The tasks.
            station: Per task, its station.
            start: The start decision.
        """
        is_clean = model.param(lambda task: task.is_clean, tasks)
        product = model.param(lambda task: task.product, tasks)
        cleaning = Transition(
            kind=product,
            same=model.param(lambda task: task.clean_gap_same_product, tasks),
            other=model.param(lambda task: task.clean_gap_other_product, tasks),
        )
        model.no_overlap(
            station, start, name="clean_task_spacing", where=is_clean, transition=cleaning
        )

        # clean stations in use and product changes between a station's clean tasks,
        # less one per clean product
        weight = self.config.weights["clean_product_chaining"]

        for station_group in station.groups:
            clean = station_group.where(is_clean)
            model.minimize(clean.size > 0, name="clean_product_chaining", weight=weight)
            order = model.sort(clean, key=start)
            changed = product[order.previous] != product[order.current]
            model.minimize(changed, name="clean_product_chaining", weight=weight)

        for product_group in model.group_by(lambda task: task.product, tasks):
            model.minimize(
                -(product_group.where(is_clean).size > 0),
                name="clean_product_chaining",
                weight=weight,
            )

    def _post_crew_soft_period(
        self, model: Model[Workshop], crews: SubsetFamily[Task], start: IntegerDecision
    ) -> None:
        """Time a crew works inside the periods it would rather keep free.

        Args:
            model: The model.
            crews: Per crew, its tasks.
            start: The start decision.
        """
        periods = model.set(
            lambda s: [period for task in s.tasks for period in task.crew_soft_periods],
            distinct=True,
        )
        period_start = model.param(lambda period: period.start, periods)
        period_end = model.param(lambda period: period.end, periods)
        periods_of = model.group_by(lambda period: period.crew_id, periods, same_order_as=crews)

        for crew in crews:
            period = periods_of[crew.key].member
            task = crew.member
            overlap = model.smallest(period_end[period], start.end(task)) - model.largest(
                period_start[period], start[task]
            )

            model.minimize(
                model.sum(period, model.largest(0, overlap)),
                name="crew_soft_period",
                weight=self.config.weights["crew_soft_period"],
            )

    def _post_crew_station_changes(
        self,
        model: Model[Workshop],
        crews: SubsetFamily[Task],
        start: IntegerDecision,
        station: Assignment[Task],
    ) -> None:
        """Times a crew changes stations between two tasks of one day.

        Args:
            model: The model.
            crews: Per crew, its tasks.
            start: The start decision.
            station: Per task, its station.
        """
        for crew in crews:
            order = model.sort(crew, key=start)
            previous, current = order.previous, order.current
            same_day = self._day(start, previous) == self._day(start, current)

            model.minimize(
                same_day & (station[previous] != station[current]),
                name="crew_station_changes",
                weight=self.config.weights["crew_station_changes"],
            )

    def _post_station_utilization(self, model: Model[Workshop], station: Assignment[Task]) -> None:
        """Stations in use.

        Args:
            model: The model.
            station: Per task, its station.
        """
        for station_group in station.groups:
            model.minimize(
                station_group.size > 0,
                name="station_utilization",
                weight=self.config.weights["station_utilization"],
            )

    def _post_station_preference(
        self,
        model: Model[Workshop],
        tasks: IndexSet[Task],
        stations: IndexSet[Station],
        station: Assignment[Task],
    ) -> None:
        """Rank of the first preferred pool holding the task's station (pool count if none).

        Args:
            model: The model.
            tasks: The tasks.
            stations: The stations.
            station: Per task, its station.
        """
        pools = model.set(lambda s: [pool for task in s.tasks for pool in task.preferred_stations])
        pools_of = model.subsets(lambda task: task.preferred_stations, tasks, of=pools)
        stations_of = model.subsets(lambda pool: pool, pools, of=stations)

        task = tasks.member
        pool, earlier_pool = pools_of[task].member, pools_of[task].member
        pool_station = stations_of[pool].member
        held = model.count(pool_station, station[task] == pool_station) > 0

        rank = model.min(
            pool,
            model.if_then_else(
                held, model.count(earlier_pool, earlier_pool < pool), self.NOT_HELD
            ),
        )

        model.minimize(
            model.smallest(pools_of[task].size, rank),
            name="station_preference",
            weight=self.config.weights["station_preference"],
        )
