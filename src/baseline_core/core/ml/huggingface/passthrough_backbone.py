"""Backbone handing the inputs to a Hugging Face head that owns the whole model."""

from typing import Any, override

import torch

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.model.abstract_backbone import AbstractBackbone


class PassthroughBackbone(AbstractBackbone[Batch[Any], dict[str, torch.Tensor]]):
    """No shared encoder: the inputs of the batch are the features."""

    @override
    def forward(self, batch: Batch[Any]) -> dict[str, torch.Tensor]:
        """Return the inputs of ``batch``."""
        return batch.x
