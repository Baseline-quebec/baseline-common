"""Fixtures for the training loop: data modules and the Lightning module with logging patched."""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from torch import nn
from torchmetrics import MeanSquaredError, MetricCollection

from baseline_core.core.ml.data.stacking_collator import StackingCollator, StackingCollatorConfig
from baseline_core.core.ml.model.head_container import HeadContainer
from baseline_core.core.ml.training.backbone_heads_module import BackboneHeadsModule
from baseline_core.core.ml.training.datamodule import DataModuleConfig
from baseline_core.core.ml.training.optimizer import OptimizerFactory
from tests.core.ml.dummy_classes import DummyBackbone
from tests.core.ml.training.dummy_classes import (
    GroupedSampleDataModule,
    GroupedSampleDataModuleConfig,
    RangeDataModule,
)


@pytest.fixture
def batch_size() -> int:
    """Samples per batch."""
    return 2


@pytest.fixture
def datamodule_config(batch_size: int) -> DataModuleConfig:
    """Loading parameters with the default split."""
    return DataModuleConfig(batch_size=batch_size)


@pytest.fixture
def datamodule(datamodule_config: DataModuleConfig) -> RangeDataModule:
    """Data module over fixed integer splits, not yet set up."""
    return RangeDataModule(datamodule_config)


@pytest.fixture
def metrics() -> dict[str, MetricCollection]:
    """One mean squared error per dummy head target."""
    return {
        "y_user": MetricCollection([MeanSquaredError()]),
        "y_flight": MetricCollection([MeanSquaredError()]),
    }


@pytest.fixture
def bare_module(
    backbone: DummyBackbone,
    heads: HeadContainer,
    optimizer_factory: OptimizerFactory,
    metrics: dict[str, MetricCollection],
) -> BackboneHeadsModule:
    """Lightning module as built, before its logging is patched."""
    return BackboneHeadsModule(backbone, heads, optimizer_factory, metrics)


@pytest.fixture
def log(bare_module: BackboneHeadsModule, mocker: MockerFixture) -> MagicMock:
    """``module.log`` replaced by a mock: there is no trainer to receive the values."""
    return mocker.patch.object(bare_module, "log")


@pytest.fixture
def log_dict(bare_module: BackboneHeadsModule, mocker: MockerFixture) -> MagicMock:
    """``module.log_dict`` replaced by a mock, for the same reason as ``log``."""
    return mocker.patch.object(bare_module, "log_dict")


@pytest.fixture
def module(
    bare_module: BackboneHeadsModule, log: MagicMock, log_dict: MagicMock
) -> BackboneHeadsModule:
    """Lightning module with ``log`` and ``log_dict`` patched; assert on those fixtures."""
    del log, log_dict
    return bare_module


@pytest.fixture
def grouped_datamodule(hidden_dim: int) -> GroupedSampleDataModule:
    """Random splits for the dummy model, stacked into batches."""
    collator = StackingCollator(
        StackingCollatorConfig(
            group_mapping={"x": ["user", "flight"], "targets": ["y_user", "y_flight"]}
        )
    )
    return GroupedSampleDataModule(
        GroupedSampleDataModuleConfig(batch_size=2, sample_count=4, feature_count=hidden_dim),
        collator,
    )


@pytest.fixture
def parameters() -> list[nn.Parameter]:
    """Parameters of a tiny layer."""
    return list(nn.Linear(2, 1).parameters())
