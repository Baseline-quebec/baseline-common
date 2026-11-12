"""Small models over the crew workshop, each built to exercise one behavior of the engine."""

from collections.abc import Sequence

from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.model import Model
from tests.core.optimization.crew_model import (
    Alternatives,
    CrewModelConfig,
    CrewModelDefinition,
    Route,
    Station,
    Step,
    Task,
    TaskKind,
    Tasks,
    Window,
    Workshop,
)


class WorkshopFactory:
    """Builds the tasks and workshops of the tests, one window per day a task may take."""

    def __init__(self, config: CrewModelConfig) -> None:
        """Keep the settings that give the days.

        Args:
            config: The settings of the workshop rules.
        """
        self.config = config

    def task(
        self,
        task_id: str,
        *,
        days: Sequence[int] = (0,),
        hours: list[int] | None = None,
        **fields: object,
    ) -> Task:
        """A task open on ``days`` (Monday by default), its own crew, order and route.

        Args:
            task_id: Its id, which also names its route.
            days: The days it may take.
            hours: The ``[first, end)`` hours it may take each day, ``[8, 18]`` by default.
            fields: Any other field of the task.

        Returns:
            The task.
        """
        first, end = hours or [8, 18]
        slots = self.config.slots_per_day
        windows = [Window(start=day * slots + first, end=day * slots + end) for day in days]

        order_key = str(fields.get("order_key", f"O_{task_id}"))
        defaults = {
            "product": f"P_{order_key}",
            "crew_id": f"C_{task_id}",
            "kind": TaskKind.STANDARD,
            "order_key": order_key,
        }
        return Task.model_validate({
            "id_": task_id,
            "name": task_id,
            "route_id": task_id,
            "windows": windows,
            **defaults,
            **fields,
        })

    @staticmethod
    def workshop(
        tasks: list[Task],
        stations: list[list[str]],
        durations: list[int],
        capacities: dict[str, int] | None = None,
    ) -> Workshop:
        """A workshop of ``tasks``, each routed to its stations for its duration.

        Args:
            tasks: The tasks.
            stations: Per task, the stations it may take.
            durations: Per task, how long it lasts.
            capacities: Per station, how many tasks it holds at once (1 when not given).

        Returns:
            The workshop.
        """
        routes = []

        for task, task_stations, duration in zip(tasks, stations, durations, strict=True):
            steps = [
                Step.model_validate({
                    "id_": f"S_{station}",
                    "name": station,
                    "station_id": station,
                    "duration": duration,
                })
                for station in task_stations
            ]
            first: list[Step | Alternatives] = (
                [*steps] if len(steps) == 1 else [Alternatives(options=steps)]
            )
            routes.append(Route(id_=task.route_id, name=task.route_id, steps=first))

        station_ids = sorted({station for task_stations in stations for station in task_stations})

        return Workshop(
            tasks=Tasks(list(tasks)),
            stations=[
                Station(id_=station, name=station, capacity=(capacities or {}).get(station, 1))
                for station in station_ids
            ],
            routes=routes,
        )


class SharedStationDefinition(AbstractModelDefinition[Workshop]):
    """Tasks in their windows and stations; a station holds as many tasks as its capacity."""

    schema = Workshop

    def __init__(self, config: CrewModelConfig) -> None:
        """Keep the settings that give the horizon.

        Args:
            config: The settings of the workshop rules.
        """
        self.config = config

    def define(self, model: Model[Workshop]) -> None:
        """Post the windows, the station options and the station capacity.

        Args:
            model: The model to post on.
        """
        tasks = model.set(lambda s: s.tasks, key="id_")
        stations = model.set(lambda s: s.stations, key="id_")
        routes = model.set(lambda s: s.routes, key="id_")

        start = model.var(
            "start",
            tasks,
            span=model.param(lambda task: routes[task.route_id].options[0].duration, tasks),
            high=self.config.n_days * self.config.slots_per_day,
        )

        options = model.subsets(
            lambda task: [option.station_id for option in routes[task.route_id].options],
            tasks,
            of=stations,
        )
        station = model.assign("station", tasks, stations, among=options)

        CrewModelDefinition.post_window_coverage(model, tasks, start)

        model.cumulative(
            station,
            start,
            demand=1,
            capacity=model.param(lambda each: each.capacity, stations),
            name="station_capacity",
        )


class DistinctStartDefinition(AbstractModelDefinition[Workshop]):
    """The tasks of one crew may not start at the same slot, yet may overlap: no no-overlap."""

    schema = Workshop

    def __init__(self, config: CrewModelConfig) -> None:
        """Keep the settings that give the horizon.

        Args:
            config: The settings of the workshop rules.
        """
        self.config = config

    def define(self, model: Model[Workshop]) -> None:
        """Post the windows and the distinct starts.

        Args:
            model: The model to post on.
        """
        tasks = model.set(lambda s: s.tasks, key="id_")
        routes = model.set(lambda s: s.routes, key="id_")

        start = model.var(
            "start",
            tasks,
            span=model.param(lambda task: routes[task.route_id].options[0].duration, tasks),
            high=self.config.n_days * self.config.slots_per_day,
        )

        CrewModelDefinition.post_window_coverage(model, tasks, start)

        for crew in model.group_by(lambda task: task.crew_id, tasks):
            task, other = crew.pair
            model.constraint(start[task] != start[other], name="distinct_start")
