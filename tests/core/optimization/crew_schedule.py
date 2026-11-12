"""The crew workshop's output schema: what a solution of the crew model reads as."""

from pydantic import AliasPath, BaseModel, ConfigDict, Field, computed_field

from tests.core.optimization.crew_model import Reservation, TaskKind


class PlacedTask(BaseModel):
    """One task, where the solution put it: read off one ``assigned`` entry."""

    # the pydantic mypy plugin types an AliasPath field only when its name validates too
    model_config = ConfigDict(validate_by_name=True)

    task_id: str = Field(validation_alias=AliasPath("element", "id_"))
    order_key: str = Field(validation_alias=AliasPath("element", "order_key"))
    kind: TaskKind = Field(validation_alias=AliasPath("element", "kind"))
    station_id: str = Field(validation_alias=AliasPath("target", "id_"))
    start_slot: int = Field(validation_alias="value")
    end_slot: int = Field(validation_alias="end")
    reservation: Reservation | None = Field(validation_alias=AliasPath("element", "reservation"))
    """The start, maybe the station, held for it; ``None`` when none is held."""

    @computed_field  # type: ignore[prop-decorator]  # pydantic's documented way, mypy disagrees
    @property
    def at_reservation(self) -> bool | None:
        """Whether the task sits at its held start and station; ``None`` if none is held."""
        if self.reservation is None:
            return None

        at_start = self.start_slot == self.reservation.absolute_start
        return at_start and self.reservation.station_id in {None, self.station_id}


class CrewSchedule(BaseModel):
    """The crew workshop's solution: the tasks placed, and the reservations they keep."""

    tasks: list[PlacedTask] = Field(validation_alias="assigned")

    @computed_field  # type: ignore[prop-decorator]  # pydantic's documented way, mypy disagrees
    @property
    def reservations(self) -> dict[str, int]:
        """How many tasks hold a reservation, and how many sit at it."""
        reserved = [task for task in self.tasks if task.reservation is not None]
        return {"total": len(reserved), "kept": sum(bool(t.at_reservation) for t in reserved)}
