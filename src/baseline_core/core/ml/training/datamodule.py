"""Lightning DataModules: the skeleton shared by every task and the one over a data source."""

from abc import ABC, abstractmethod
from typing import Any, Self, cast

import pytorch_lightning as pl
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset, random_split

from baseline_core.core.ml.data.abstract_collator import AbstractCollator
from baseline_core.core.ml.data.abstract_data_source import AbstractDataSource
from baseline_core.core.ml.data.base_dataset import BaseDataset, BaseDatasetConfig
from baseline_core.core.ml.data.cache import AbstractTransformCache
from baseline_core.core.ml.data.phase import Phase


class DataModuleConfig(BaseModel):
    """How the samples are split into phases and loaded in batches.

    Attributes:
        batch_size: Samples per batch.
        val_ratio: Fraction of the samples held out for validation; ``0`` for none.
        test_ratio: Fraction of the samples held out for test; ``0`` for none.
        seed: Seed of the shuffle, so the same samples always split the same way.
        num_workers: DataLoader worker processes.
        pin_memory: Pin host memory for faster host-to-device copies.
    """

    batch_size: int = Field(gt=0)
    val_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    seed: int = 0
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def validate_training_share(self) -> Self:
        """Validation and test must leave samples for training."""
        if self.val_ratio + self.test_ratio >= 1.0:
            error_msg = (
                f"val_ratio + test_ratio must be below 1 to leave training data, "
                f"got {self.val_ratio} + {self.test_ratio}."
            )
            raise ValueError(error_msg)

        return self

    @property
    def ratios(self) -> dict[Phase, float]:
        """Fraction of the phases that get samples, training taking the rest."""
        ratios = {
            Phase.TRAIN: 1.0 - self.val_ratio - self.test_ratio,
            Phase.VAL: self.val_ratio,
            Phase.TEST: self.test_ratio,
        }
        return {phase: ratio for phase, ratio in ratios.items() if ratio > 0}


class AbstractDataModule[S_Dataset: Dataset[Any]](pl.LightningDataModule, ABC):
    """Prepare the train/val/test datasets and expose their loaders to Lightning.

    Attributes:
        config: Loading parameters.
        collator: Abstract collator that turns a batch of samples into model inputs, or ``None``.
    """

    def __init__(self, config: DataModuleConfig, collator: AbstractCollator | None = None) -> None:
        """Store the configuration; datasets are created later in ``setup``.

        Args:
            config: Loading parameters.
            collator: ``collate_fn`` of the loaders.
        """
        super().__init__()
        self.config = config
        self.collator = collator
        self.train_data: S_Dataset | None = None
        self.val_data: S_Dataset | None = None
        self.test_data: S_Dataset | None = None

    @abstractmethod
    def prepare_data(self) -> None:
        """Download or preprocess data once, from a single process."""

    @abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Split the data and assign ``train_data``, ``val_data`` and ``test_data``.

        Args:
            stage: Lightning stage (``"fit"``, ``"validate"``, ``"test"`` or ``"predict"``).
        """

    def split_indices(self, sample_count: int) -> dict[Phase, list[int]]:
        """Partition the sample indices at random per phase, following the configured ratios.

        Args:
            sample_count: Number of samples; the same count and seed give the same partition.

        Returns:
            The indices of every phase with a non-zero ratio.

        Raises:
            ValueError: Naming the phases whose share rounds to no sample.
        """
        ratios = self.config.ratios
        subsets = random_split(
            cast("Dataset[int]", range(sample_count)),
            list(ratios.values()),
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        indices = {
            phase: list(subset.indices) for phase, subset in zip(ratios, subsets, strict=True)
        }
        empty = [phase.value for phase, phase_indices in indices.items() if not phase_indices]

        if empty:
            error_msg = (
                f"Splits without any sample: {empty} out of {sample_count}; "
                "use more samples or other ratios."
            )
            raise ValueError(error_msg)

        return indices

    def get_loader(self, dataset: S_Dataset, phase: Phase) -> DataLoader[Any]:
        """Build the loader of ``dataset`` for ``phase``; override for custom loaders.

        Args:
            dataset: Dataset to iterate.
            phase: Phase the loader serves; only training shuffles.

        Returns:
            A DataLoader using the module's collator.
        """
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=phase is Phase.TRAIN,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collator,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Loader over ``train_data``."""
        return self.get_loader(self._require(self.train_data, Phase.TRAIN), Phase.TRAIN)

    def val_dataloader(self) -> DataLoader[Any] | list[DataLoader[Any]]:
        """Loader over ``val_data``, or no loader at all when the task has no validation split."""
        return self._optional_loader(self.val_data, Phase.VAL)

    def test_dataloader(self) -> DataLoader[Any] | list[DataLoader[Any]]:
        """Loader over ``test_data``, or no loader at all when the task has no test split."""
        return self._optional_loader(self.test_data, Phase.TEST)

    def _optional_loader(
        self, dataset: S_Dataset | None, phase: Phase
    ) -> DataLoader[Any] | list[DataLoader[Any]]:
        """Loader over ``dataset``, or an empty list, which tells Lightning to skip the phase."""
        return [] if dataset is None else self.get_loader(dataset, phase)

    @staticmethod
    def _require(dataset: S_Dataset | None, phase: Phase) -> S_Dataset:
        """Return ``dataset``, failing if ``setup`` has not built it.

        Raises:
            RuntimeError: If ``dataset`` is ``None``.
        """
        if dataset is None:
            error_msg = f"{phase.value}_data is not set. Call setup() first."
            raise RuntimeError(error_msg)

        return dataset


class SourceDataModuleConfig(DataModuleConfig):
    """Split and loading parameters, plus the feature groups of the datasets the module builds.

    Attributes:
        dataset_config: Feature groups of the ``BaseDataset`` built over the source.
    """

    dataset_config: BaseDatasetConfig


class SourceDataModule(AbstractDataModule[BaseDataset]):
    """Split a data source and serve one ``BaseDataset`` per phase.

    Attributes:
        config: Split, loading parameters and feature groups.
        source: Every sample, before the split.
        cache: Results of the cached transform steps, shared by every phase, or ``None``.
    """

    config: SourceDataModuleConfig

    def __init__(
        self,
        config: SourceDataModuleConfig,
        source: AbstractDataSource[Any],
        collator: AbstractCollator | None = None,
        cache: AbstractTransformCache | None = None,
    ) -> None:
        """Store the configuration and the source; datasets are created later in ``setup``.

        Args:
            config: Split, loading parameters and feature groups.
            source: Every sample, before the split.
            collator: ``collate_fn`` of the loaders.
            cache: Results of the cached transform steps, or ``None``.
        """
        super().__init__(config, collator)
        self.source = source
        self.cache = cache

    def prepare_data(self) -> None:
        """Nothing to fetch by default; override to download or sync data once per node."""

    def setup(self, stage: str | None = None) -> None:
        """Split the source and assign one dataset per phase with a non-zero ratio.

        Args:
            stage: Ignored; every phase is built.
        """
        del stage
        dataset = BaseDataset(self.source, self.config.dataset_config, cache=self.cache)
        phase_datasets = {
            phase: dataset.select(indices, training=phase is Phase.TRAIN)
            for phase, indices in self.split_indices(len(dataset)).items()
        }
        self.train_data = phase_datasets[Phase.TRAIN]
        self.val_data = phase_datasets.get(Phase.VAL)
        self.test_data = phase_datasets.get(Phase.TEST)
