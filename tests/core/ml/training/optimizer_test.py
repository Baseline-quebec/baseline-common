"""Tests for the optimizer and scheduler factories."""

from typing import ClassVar

import pytest
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline_core.core.ml.training.optimizer import (
    AdamOptimizer,
    AdamWOptimizer,
    OptimizerFactory,
    PlateauScheduler,
)


class TestAdamOptimizer:
    """Tests for ``AdamOptimizer`` and ``AdamWOptimizer``."""

    LR: ClassVar[float] = 0.01
    WEIGHT_DECAY: ClassVar[float] = 0.1

    @pytest.fixture
    def adam(self) -> AdamOptimizer:
        """Adam without AMSGrad."""
        return AdamOptimizer(lr=self.LR, amsgrad=False)

    @pytest.fixture
    def adamw(self) -> AdamWOptimizer:
        """AdamW with weight decay."""
        return AdamWOptimizer(lr=self.LR, weight_decay=self.WEIGHT_DECAY)

    def test__call__should_build_adam_with_config(
        self, adam: AdamOptimizer, parameters: list[nn.Parameter]
    ) -> None:
        """Hyperparameters reach the optimizer."""
        optimizer = adam(parameters)

        assert isinstance(optimizer, Adam)
        assert optimizer.defaults["lr"] == self.LR
        assert optimizer.defaults["amsgrad"] is False

    def test__call__when_adamw__should_build_adamw_with_weight_decay(
        self, adamw: AdamWOptimizer, parameters: list[nn.Parameter]
    ) -> None:
        """AdamW adds decoupled weight decay."""
        optimizer = adamw(parameters)

        assert isinstance(optimizer, AdamW)
        assert optimizer.defaults["weight_decay"] == self.WEIGHT_DECAY

    def test__init__when_lr_not_positive__should_raise(self) -> None:
        """A zero learning rate is rejected."""
        with pytest.raises(ValueError, match="lr"):
            AdamOptimizer(lr=0.0)


class TestPlateauScheduler:
    """Tests for ``PlateauScheduler``."""

    @pytest.fixture
    def scheduler(self) -> PlateauScheduler:
        """Plateau scheduler maximizing the validation loss."""
        return PlateauScheduler(metric="val_loss", mode="max")

    @pytest.fixture
    def optimizer(self, parameters: list[nn.Parameter]) -> torch.optim.Optimizer:
        """Plain SGD over ``parameters``."""
        return torch.optim.SGD(parameters, lr=0.1)

    def test__call__should_return_lightning_scheduler_config(
        self, scheduler: PlateauScheduler, optimizer: torch.optim.Optimizer
    ) -> None:
        """The result is the dict Lightning expects under ``lr_scheduler``."""
        config = scheduler(optimizer)

        assert isinstance(config["scheduler"], ReduceLROnPlateau)
        assert config["scheduler"].mode == "max"
        assert config["monitor"] == "val_loss"
        assert config["interval"] == "epoch"


class TestOptimizerFactory:
    """Tests for ``OptimizerFactory``."""

    @pytest.fixture
    def plain_factory(self) -> OptimizerFactory:
        """Adam without a scheduler."""
        return OptimizerFactory(optimizer=AdamOptimizer(lr=0.01))

    def test__call__when_scheduler__should_return_both(
        self, optimizer_factory: OptimizerFactory, parameters: list[nn.Parameter]
    ) -> None:
        """Optimizer and scheduler are returned together."""
        result = optimizer_factory(parameters)

        assert isinstance(result["optimizer"], Adam)
        assert "lr_scheduler" in result

    def test__call__when_no_scheduler__should_return_optimizer_only(
        self, plain_factory: OptimizerFactory, parameters: list[nn.Parameter]
    ) -> None:
        """Without a scheduler, only the optimizer is returned."""
        result = plain_factory(parameters)

        assert set(result) == {"optimizer"}
