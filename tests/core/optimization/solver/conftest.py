"""Fixtures of the solver tests: a crew's tasks, a small job shop, an overloaded fleet."""

import pytest

from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel, ModelCompiler
from tests.core.optimization.bin_packing_model import Fleet, Parcel, StackingModelDefinition, Truck
from tests.core.optimization.crew_model import Workshop
from tests.core.optimization.job_shop_model import (
    EarlyStartModelDefinition,
    JobShopModelDefinition,
    Shop,
)
from tests.core.optimization.models import WorkshopFactory


@pytest.fixture
def bound(compiled: ModelCompiler[Workshop], factory: WorkshopFactory) -> BoundModel:
    """Return three two-hour tasks of one crew on Monday, two stations each, at their defaults.

    Args:
        compiled: The compiled crew model.
        factory: The test factory.

    Returns:
        The bound model, placed at its defaults.
    """
    tasks = [factory.task(name, crew_id="C") for name in ["a", "b", "c"]]
    bound = compiled.bind(factory.workshop(tasks, [["S1", "S2"]] * 3, [2, 2, 2]))
    bound.place([])
    return bound


@pytest.fixture(scope="session")
def stacking() -> ModelCompiler[Fleet]:
    """Return the stacking model, compiled once.

    Returns:
        The compiled model.
    """
    return ModelCompiler(StackingModelDefinition(), cythonize=False)


@pytest.fixture(scope="session")
def compiled_stacking() -> ModelCompiler[Fleet]:
    """Return the stacking model, compiled once to a native kernel.

    Returns:
        The compiled model.
    """
    return ModelCompiler(StackingModelDefinition(), cythonize=True)


@pytest.fixture
def fleet() -> Fleet:
    """Return nine parcels and three trucks; together the parcels weigh more than any bears.

    Returns:
        The fleet.
    """
    parcels = [
        Parcel(ref=f"p{i}", weight=1 + i % 3, trucks=["A", "B", "C"], bears=5 + i % 3)
        for i in range(9)
    ]
    return Fleet(
        trucks=[Truck(code=code, capacity=0) for code in ["A", "B", "C"]], parcels=parcels
    )


@pytest.fixture
def overloaded(stacking: ModelCompiler[Fleet], fleet: Fleet) -> BoundModel:
    """Return the fleet with every parcel in the first truck, which none of them bears.

    Args:
        stacking: The compiled stacking model.
        fleet: The fleet.

    Returns:
        The bound model, every parcel in truck A.
    """
    bound = stacking.bind(fleet)
    bound.kernel.set_placement([0] * 9, [0] * 9)
    return bound


@pytest.fixture(scope="session")
def job_shop() -> ModelCompiler[Shop]:
    """Return the job shop model, compiled once.

    Returns:
        The compiled model.
    """
    return ModelCompiler(JobShopModelDefinition(), cythonize=False)


@pytest.fixture
def shop(job_shop: ModelCompiler[Shop]) -> BoundModel:
    """Return four operations of two jobs on two machines, placed at their defaults.

    Args:
        job_shop: The compiled job shop model.

    Returns:
        The bound model.
    """
    bound = job_shop.bind(
        Shop.of([
            ["o1", "J1", 0, 3, ["M1", "M2"], 0, 12],
            ["o2", "J1", 1, 2, ["M1", "M2"], 0, 12],
            ["o3", "J2", 0, 4, ["M1", "M2"], 0, 12],
            ["o4", "J2", 1, 3, ["M1", "M2"], 0, 12],
        ])
    )
    bound.place([])
    return bound


@pytest.fixture
def late() -> BoundModel:
    """Return two operations at 0 on one machine, where an hour of delay costs a million.

    Returns:
        The bound model, the second operation overlapping the first.
    """
    shop = Shop.of([["a", "J1", 0, 2, ["M1"], 0, 40], ["b", "J2", 0, 2, ["M1"], 0, 40]])
    bound = ModelCompiler(EarlyStartModelDefinition(), cythonize=False).bind(shop)
    bound.kernel.load(start=[0, 0], machine=[0, 0])
    return bound
