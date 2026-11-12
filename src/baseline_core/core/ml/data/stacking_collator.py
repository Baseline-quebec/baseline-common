"""Collator stacking or padding sample keys into a batch."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from pydantic import BaseModel, Field
from torch.nn.utils.rnn import pad_sequence

from baseline_core.core.ml.data.abstract_collator import AbstractCollator
from baseline_core.core.ml.data.batch import Batch


class StackingCollatorConfig(BaseModel):
    """How ``StackingCollator`` groups sample keys into a batch.

    Attributes:
        group_mapping: Batch field to the sample keys it gathers, e.g. ``{"x": ["image"]}``.
        keys_to_pad: Sample keys padded to a common length instead of stacked.
        padding_value: Value used for padding.
    """

    group_mapping: dict[str, list[str]]
    keys_to_pad: list[str] = Field(default_factory=list)
    padding_value: float = 0.0


class StackingCollator(AbstractCollator):
    """Group sample keys per ``group_mapping``, stack or pad them, build a ``Batch``.

    Attributes:
        config: Grouping and padding rules.
        batch_factory: Builds the batch from the grouped fields; a ``Batch`` subclass by default.
    """

    def __init__(
        self, config: StackingCollatorConfig, batch_factory: Callable[..., Batch[Any]] = Batch
    ) -> None:
        """Store the configuration and the batch factory.

        Args:
            config: Grouping and padding rules.
            batch_factory: Builds the batch from the grouped fields, e.g. a ``Batch`` subclass.
        """
        self.config = config
        self.batch_factory = batch_factory

    def __call__(self, samples: list[dict[str, Any]]) -> Batch[Any]:
        """Collate ``samples`` into one batch.

        Args:
            samples: Outputs of ``Dataset.__getitem__``.

        Returns:
            The batch built by ``batch_factory`` from fields following ``config.group_mapping``.
        """
        transposed: dict[str, list[Any]] = defaultdict(list)

        for sample in samples:
            for key, value in sample.items():
                transposed[key].append(value)

        batch_kwargs: dict[str, dict[str, Any]] = {}

        for group_name, keys in self.config.group_mapping.items():
            group = {
                key: self.collate_field(key, transposed[key]) for key in keys if key in transposed
            }

            if group:
                batch_kwargs[group_name] = group

        return self.batch_factory(**batch_kwargs)

    def collate_field(self, key: str, values: list[Any]) -> Any:  # noqa: ANN401
        """Aggregate the values of one key across the batch.

        Args:
            key: Sample key being collated.
            values: One value per sample.

        Returns:
            Stacked (or padded) tensors, a tensor of numbers, or a list of anything else.
        """
        if key in self.config.keys_to_pad:
            return pad_sequence(values, batch_first=True, padding_value=self.config.padding_value)

        first = values[0]

        if isinstance(first, torch.Tensor):
            return torch.stack(values)

        if isinstance(first, int | float):
            return torch.tensor(values)

        return values
