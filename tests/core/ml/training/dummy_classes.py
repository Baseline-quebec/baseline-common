"""Minimal data modules for the training loop tests."""

from typing import Any

import torch
from torch.utils.data import Dataset

from baseline_core.core.ml.training.datamodule import AbstractDataModule, DataModuleConfig


class RangeDataset(Dataset[int]):
    """Dataset over a list of integers."""

    def __init__(self, values: list[int]) -> None:
        """Store the values."""
        self.values = values

    def __len__(self) -> int:
        """Number of values."""
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        """Return the value at ``index``."""
        return self.values[index]


class RangeDataModule(AbstractDataModule[RangeDataset]):
    """Data module with fixed integer splits."""

    TRAIN_VALUES = (1, 2, 3, 4, 5)
    VAL_VALUES = (6, 7, 8)
    TEST_VALUES = (9, 10)

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Track whether ``prepare_data`` ran."""
        super().__init__(*args, **kwargs)
        self.prepared = False

    def prepare_data(self) -> None:
        """Record the call."""
        self.prepared = True

    def setup(self, stage: str | None = None) -> None:
        """Assign the splits of ``stage`` (all of them when ``None``)."""
        if stage in {"fit", None}:
            self.train_data = RangeDataset(list(self.TRAIN_VALUES))
            self.val_data = RangeDataset(list(self.VAL_VALUES))

        if stage in {"test", None}:
            self.test_data = RangeDataset(list(self.TEST_VALUES))


class GroupedSampleDataset(Dataset[dict[str, torch.Tensor]]):
    """Random samples of the dummy model: a ``user`` and a ``flight`` vector and their targets."""

    def __init__(self, sample_count: int, feature_count: int) -> None:
        """Draw the samples."""
        self.samples = [
            {
                "user": torch.randn(feature_count),
                "flight": torch.randn(feature_count),
                "y_user": torch.randn(1),
                "y_flight": torch.rand(1).round(),
            }
            for _ in range(sample_count)
        ]

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Features and targets of sample ``index``."""
        return self.samples[index]


class GroupedSampleDataModuleConfig(DataModuleConfig):
    """Size of the splits.

    Attributes:
        sample_count: Samples per split.
        feature_count: Features per vector.
    """

    sample_count: int
    feature_count: int


class GroupedSampleDataModule(AbstractDataModule[GroupedSampleDataset]):
    """Random splits for the dummy model."""

    config: GroupedSampleDataModuleConfig

    def prepare_data(self) -> None:
        """Nothing to prepare."""

    def setup(self, stage: str | None = None) -> None:
        """Draw every split."""
        del stage
        self.train_data, self.val_data, self.test_data = (
            GroupedSampleDataset(self.config.sample_count, self.config.feature_count)
            for _ in range(3)
        )
