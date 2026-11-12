"""Crew workshops of real size, drawn at random."""

from typing import ClassVar

from numpy.random import default_rng

from tests.core.optimization.crew_model import Workshop


class LargeCrewProblem:
    """Crews of ~10 tasks (2-3 h), stations in pools of 20, clean ones in pools of 5, co-starts."""

    WEEK: ClassVar[list[dict[str, int]]] = [
        {"start": day * 24 + 8, "end": day * 24 + 18} for day in range(5)
    ]
    """Every task's windows: Monday to Friday, 8 h to 18 h."""
    CLEAN_SHARE: ClassVar[float] = 0.25
    """The share of tasks that need a clean station."""

    def __init__(self, n_crews: int) -> None:
        """Keep the size of the workshops drawn.

        Args:
            n_crews: The number of crews.
        """
        self.n_crews = n_crews

    def build(self, seed: int) -> Workshop:  # noqa: PLR0914  one draw, in order
        """Draw one workshop.

        Args:
            seed: The draw.

        Returns:
            The workshop.
        """
        rng = default_rng(seed)
        tasks: list[dict[str, object]] = []
        routes: list[dict[str, object]] = []
        n_tasks_guess = self.n_crews * 10
        standard = [f"S{k}" for k in range(max(n_tasks_guess // 25, 20))]
        clean = [f"K{k}" for k in range(max(n_tasks_guess // 100, 5))]

        for crew in range(self.n_crews):
            pool = int(rng.integers(0, len(standard) // 20))
            clean_pool = int(rng.integers(0, len(clean) // 5))

            for c in range(int(rng.integers(3, 6))):
                product, spacing = f"P{crew}_{c}", str(rng.choice(["none", "none", "day_apart"]))

                for run in range(1, int(rng.integers(2, 4))):
                    is_clean = rng.random() < self.CLEAN_SHARE
                    stations = (
                        clean[clean_pool * 5 : clean_pool * 5 + 5]
                        if is_clean
                        else standard[pool * 20 : pool * 20 + 20]
                    )

                    task, duration = f"{product}_task_{run}", int(rng.choice([2, 3]))
                    tasks.append({
                        "id_": task,
                        "name": task,
                        "route_id": task,
                        "windows": self.WEEK,
                        "product": product,
                        "crew_id": f"C{crew}",
                        "kind": "C" if is_clean else "S",
                        "order_key": f"{product}_001",
                        "run": run,
                        "spacing": spacing,
                        "clean_gap_same_product": 1,
                        "clean_gap_other_product": 2,
                    })

                    options = [
                        {
                            "id_": f"{task}_at_{station}",
                            "name": f"{task} at {station}",
                            "station_id": station,
                            "duration": duration,
                        }
                        for station in stations
                    ]
                    routes.append({"id_": task, "name": task, "steps": [{"options": options}]})

        crew_of = {task["id_"]: task["crew_id"] for task in tasks}
        ids = list(crew_of)
        co_start = [
            [first, second]
            for first, second in zip(ids[::40], ids[20::40], strict=False)
            if crew_of[first] != crew_of[second]
        ]

        return Workshop.model_validate({
            "tasks": tasks,
            "stations": [{"id_": station, "name": station} for station in standard + clean],
            "routes": routes,
            "co_start_groups": co_start,
        })
