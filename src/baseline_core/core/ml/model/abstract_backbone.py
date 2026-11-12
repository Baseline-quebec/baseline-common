"""Backbone: turns a batch into embeddings shared by every head."""

from abc import ABC, abstractmethod

from torch import nn

from baseline_core.core.ml.data.batch import AbstractBatch


class AbstractBackbone[BatchT: AbstractBatch, FeaturesT](nn.Module, ABC):
    """Feature extractor shared by every head."""

    def __call__(self, batch: BatchT) -> FeaturesT:
        """Run ``forward`` through the ``nn.Module`` hooks, with a typed result."""
        features: FeaturesT = super().__call__(batch)
        return features

    @abstractmethod
    def forward(self, batch: BatchT) -> FeaturesT:
        """Encode ``batch`` into embeddings.

        Args:
            batch: Model inputs, never modified.

        Returns:
            The embeddings every head reads.
        """
