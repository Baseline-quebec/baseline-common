"""Fixtures: the crew workshop's settings, its model compiled once, and its test factory."""

import pytest

from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.infeasibility.fork_pool import ForkPool
from tests.core.optimization.counter import Counter
from tests.core.optimization.crew_model import CrewModelConfig, CrewModelDefinition, Workshop
from tests.core.optimization.models import WorkshopFactory


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:  # noqa: ARG001  xdist's hook
    """The workers ``-n auto`` starts with pytest-xdist: every core but two.

    Args:
        config: The pytest configuration.

    Returns:
        The number of workers.
    """
    return max(ForkPool.available_cores() - 2, 1)


@pytest.fixture(scope="session")
def config() -> CrewModelConfig:
    """The default settings of the workshop rules: the week, the break hours, the spacing.

    Returns:
        The settings.
    """
    return CrewModelConfig()


@pytest.fixture(scope="session")
def compiled(config: CrewModelConfig) -> ModelCompiler[Workshop]:
    """The crew model, its kernel run as plain Python.

    Args:
        config: The settings of the workshop rules.

    Returns:
        The compiled model.
    """
    return ModelCompiler(CrewModelDefinition(config), cythonize=False)


@pytest.fixture(scope="session")
def cythonized(config: CrewModelConfig) -> ModelCompiler[Workshop]:
    """The crew model, its kernel compiled with Cython.

    Args:
        config: The settings of the workshop rules.

    Returns:
        The compiled model.
    """
    return ModelCompiler(CrewModelDefinition(config))


@pytest.fixture(scope="session")
def counter(compiled: ModelCompiler[Workshop]) -> Counter:
    """Counts per constraint of the crew model.

    Args:
        compiled: The compiled crew model.

    Returns:
        The counter.
    """
    return Counter(compiled=compiled)


@pytest.fixture(scope="session")
def factory(config: CrewModelConfig) -> WorkshopFactory:
    """Builds the tasks and workshops of the tests.

    Args:
        config: The settings of the workshop rules.

    Returns:
        The factory.
    """
    return WorkshopFactory(config)
