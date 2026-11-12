"""Parcels loaded into trucks, with no time at all: models whose only decision is a target."""

from pydantic import BaseModel

from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.model import Model


class Truck(BaseModel):
    """A truck and what it carries."""

    code: str
    capacity: int


class Parcel(BaseModel):
    """A parcel, its weight, and the trucks it may go in."""

    ref: str
    weight: int
    trucks: list[str]
    bears: int = 1_000
    """The most a truck may carry, this parcel included, with it on board."""


class Fleet(BaseModel):
    """The trucks and the parcels to load."""

    trucks: list[Truck]
    parcels: list[Parcel]


class BinPackingModelDefinition(AbstractModelDefinition[Fleet]):
    """Each parcel in one of its trucks; no truck carries more than its capacity."""

    schema = Fleet

    def define(self, model: Model[Fleet]) -> None:
        """Post the assignment and the capacity.

        Args:
            model: The model to post on.
        """
        parcels = model.set(lambda s: s.parcels, key="ref")
        trucks = model.set(lambda s: s.trucks, key="code")

        options = model.subsets(lambda parcel: parcel.trucks, parcels, of=trucks)
        truck = model.assign("truck", parcels, trucks, among=options)

        model.bin_packing(
            truck,
            demand=model.param(lambda parcel: parcel.weight, parcels),
            capacity=model.param(lambda each: each.capacity, trucks),
            name="truck_capacity",
        )


class StackingModelDefinition(AbstractModelDefinition[Fleet]):
    """Each parcel in one of its trucks; no truck carries more than any parcel on board bears."""

    schema = Fleet

    def define(self, model: Model[Fleet]) -> None:
        """Post the assignment and, per parcel, the load of its truck.

        Args:
            model: The model to post on.
        """
        parcels = model.set(lambda s: s.parcels, key="ref")
        trucks = model.set(lambda s: s.trucks, key="code")

        options = model.subsets(lambda parcel: parcel.trucks, parcels, of=trucks)
        truck = model.assign("truck", parcels, trucks, among=options)
        weight = model.param(lambda parcel: parcel.weight, parcels)
        bears = model.param(lambda parcel: parcel.bears, parcels)

        for group in truck.groups:
            parcel, other = group.member, group.member
            model.constraint(model.sum(other, weight[other]) <= bears[parcel], name="stacking")
