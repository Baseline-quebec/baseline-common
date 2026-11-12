"""Optimizer and scheduler factories built from configs."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from torch.nn import Parameter
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class AbstractOptimizer(BaseModel, ABC):
    """Config that builds an optimizer from model parameters.

    Attributes:
        lr: Learning rate.
    """

    lr: float = Field(gt=0.0)

    @abstractmethod
    def __call__(self, parameters: Iterable[Parameter]) -> Optimizer:
        """Create the optimizer for ``parameters``."""


class AbstractScheduler(BaseModel, ABC):
    """Config that builds a learning-rate scheduler for an optimizer."""

    model_config = ConfigDict(extra="allow")

    @abstractmethod
    def __call__(self, optimizer: Optimizer) -> LRScheduler | dict[str, Any]:
        """Create the scheduler for ``optimizer``, or a Lightning ``lr_scheduler`` dict."""


class OptimizerFactory(BaseModel):
    """Optimizer plus optional scheduler, in the shape ``configure_optimizers`` expects.

    Attributes:
        optimizer: Optimizer config.
        scheduler: Scheduler config, or ``None``.
    """

    optimizer: AbstractOptimizer
    scheduler: AbstractScheduler | None = None

    def __call__(self, parameters: Iterable[Parameter]) -> dict[str, Any]:
        """Build ``{"optimizer": ..., "lr_scheduler": ...}`` for ``parameters``."""
        optimizer = self.optimizer(parameters)
        result: dict[str, Any] = {"optimizer": optimizer}

        if self.scheduler is not None:
            result["lr_scheduler"] = self.scheduler(optimizer)

        return result


class PlateauScheduler(AbstractScheduler):
    """Multiply the learning rate by ``factor`` when ``metric`` stops improving.

    Attributes:
        metric: Logged metric to monitor.
        mode: Whether the metric should decrease (``"min"``) or increase (``"max"``).
        factor: Learning-rate multiplier applied on plateau.
        patience: Epochs without improvement before reducing.
    """

    metric: str
    mode: Literal["min", "max"] = "min"
    factor: float = Field(default=0.1, gt=0.0, lt=1.0)
    patience: int = Field(default=10, ge=0)

    def __call__(self, optimizer: Optimizer) -> dict[str, Any]:
        """Create the scheduler and its Lightning configuration."""
        scheduler = ReduceLROnPlateau(
            optimizer, mode=self.mode, factor=self.factor, patience=self.patience
        )
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": self.metric,
        }


class AdamOptimizer(AbstractOptimizer):
    """Adam.

    Attributes:
        betas: Coefficients for the running averages of the gradient and its square.
        eps: Term added to the denominator for numerical stability.
        amsgrad: Use the AMSGrad variant.
    """

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = True

    def __call__(self, parameters: Iterable[Parameter]) -> Adam:
        """Create the Adam optimizer."""
        return Adam(parameters, lr=self.lr, betas=self.betas, eps=self.eps, amsgrad=self.amsgrad)


class AdamWOptimizer(AdamOptimizer):
    """AdamW: Adam with decoupled weight decay.

    Attributes:
        weight_decay: Weight decay coefficient.
    """

    weight_decay: float = Field(ge=0.0)

    def __call__(self, parameters: Iterable[Parameter]) -> AdamW:
        """Create the AdamW optimizer."""
        return AdamW(
            parameters,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )
