"""Fixtures: the tiny shop compiled once, and bound to four jobs."""

import pytest

from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel, ModelCompiler
from tests.core.optimization.mining.shop import Shop, ShopDefinition


@pytest.fixture(scope="session")
def shop_compiler() -> ModelCompiler[Shop]:
    """The shop's model, its kernel run as plain Python.

    Returns:
        The compiled model.
    """
    return ModelCompiler(ShopDefinition(), cythonize=False)


@pytest.fixture
def four_jobs(shop_compiler: ModelCompiler[Shop]) -> BoundModel:
    """Bind the shop to four jobs of four operators.

    Args:
        shop_compiler: The shop's compiled model.

    Returns:
        The bound model, at the placement the bind leaves.
    """
    return shop_compiler.bind(Shop.of(["a", "b", "c", "d"]))
