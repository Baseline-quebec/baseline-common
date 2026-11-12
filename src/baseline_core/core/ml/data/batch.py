"""Batch schemas: what the collator produces and what models and losses exchange."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Self

import torch
from pydantic import BaseModel, ConfigDict


class AbstractBatch(BaseModel, ABC):
    """Tensors moved to a device together, and the number of samples they hold."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of samples in the batch."""

    def to(self, device: torch.device | str, **kwargs: Any) -> Self:  # noqa: ANN401
        """Return a copy whose tensors are on ``device``, wherever they are nested.

        Args:
            device: Destination device.
            **kwargs: Forwarded to ``torch.Tensor.to()`` (``non_blocking``, ...).

        Returns:
            The moved batch, of the same class.
        """
        return self.map_fields(lambda tensor: tensor.to(device, **kwargs))

    def pin_memory(self) -> Self:
        """Return a copy with pinned tensors, as ``DataLoader(pin_memory=True)`` asks.

        Returns:
            The pinned batch, of the same class.
        """
        return self.map_fields(torch.Tensor.pin_memory)

    def map_fields(self, operation: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        """Return a copy with ``operation`` applied to every tensor of every field.

        Args:
            operation: Tensor to tensor function.

        Returns:
            The converted batch, of the same class.
        """
        names = [*type(self).model_fields, *(self.model_extra or {})]
        return self.model_copy(
            update={name: self.map_tensors(getattr(self, name), operation) for name in names}
        )

    @staticmethod
    def map_tensors(
        value: Any,  # noqa: ANN401
        operation: Callable[[torch.Tensor], torch.Tensor],
    ) -> Any:  # noqa: ANN401
        """Apply ``operation`` to every tensor nested in dicts, lists and tuples.

        Args:
            value: Tensor or (nested) container of tensors.
            operation: Tensor to tensor function.

        Returns:
            The same structure with every tensor converted, other leaves unchanged.
        """
        if isinstance(value, torch.Tensor):
            return operation(value)

        if isinstance(value, Mapping):
            return {key: AbstractBatch.map_tensors(item, operation) for key, item in value.items()}

        if isinstance(value, list):
            return [AbstractBatch.map_tensors(item, operation) for item in value]

        if isinstance(value, tuple):
            return tuple(AbstractBatch.map_tensors(item, operation) for item in value)

        return value


class Batch[Targets](AbstractBatch):
    """Inputs the model consumes, supervision the loss consumes, and metadata.

    Attributes:
        x: Input features keyed by name, so a model can run ``model(**batch.x)``.
        targets: Supervision for the loss. ``None`` at inference time.
        metadata: Anything else the forward pass needs (sample ids, original sizes, ...).
    """

    x: dict[str, torch.Tensor]
    targets: Targets | None = None
    metadata: dict[str, Any] | None = None

    @property
    def batch_size(self) -> int:
        """Leading dimension of the first input.

        Raises:
            ValueError: If the batch has no input.
        """
        first = next(iter(self.x.values()), None)

        if first is None:
            error_msg = "A batch without inputs has no batch size."
            raise ValueError(error_msg)

        return int(first.size(0))


TensorTargets = dict[str, torch.Tensor]
