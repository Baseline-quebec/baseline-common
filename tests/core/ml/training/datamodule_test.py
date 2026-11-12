"""Tests for the data module configuration, ``AbstractDataModule`` and ``SourceDataModule``."""

from types import ModuleType
from typing import ClassVar

import pytest
from torch.utils.data import RandomSampler, SequentialSampler

from baseline_core.core.ml.data.base_dataset import BaseDatasetConfig, FeatureGroupConfig
from baseline_core.core.ml.data.list_dict_source import ListDictSource
from baseline_core.core.ml.data.phase import Phase
from baseline_core.core.ml.data.transform_pipeline import TransformConfig, TransformPipeline
from baseline_core.core.ml.training.datamodule import (
    DataModuleConfig,
    SourceDataModule,
    SourceDataModuleConfig,
)
from tests.core.ml.training.dummy_classes import RangeDataModule


class TestDataModuleConfig:
    """Tests for ``DataModuleConfig``."""

    EXPECTED_EXTRA: ClassVar[int] = 123

    def test__init__should_default_to_a_balanced_hold_out(self) -> None:
        """Without ratios, 15 % goes to validation and 15 % to test."""
        config = DataModuleConfig(batch_size=8)

        assert (config.val_ratio, config.test_ratio, config.seed) == (0.15, 0.15, 0)

    def test__ratios__should_give_the_rest_to_training(self) -> None:
        """Training takes whatever validation and test leave."""
        config = DataModuleConfig(batch_size=8, val_ratio=0.2, test_ratio=0.3)

        assert config.ratios == {Phase.TRAIN: pytest.approx(0.5), Phase.VAL: 0.2, Phase.TEST: 0.3}

    def test__ratios__when_a_ratio_is_zero__should_leave_the_phase_out(self) -> None:
        """A task without validation gets no validation phase at all."""
        config = DataModuleConfig(batch_size=8, val_ratio=0.0, test_ratio=0.2)

        assert config.ratios == {Phase.TRAIN: pytest.approx(0.8), Phase.TEST: 0.2}

    @pytest.mark.parametrize(("val_ratio", "test_ratio"), [(0.5, 0.5), (0.7, 0.4)])
    def test__init__when_ratios_leave_no_training_data__should_raise_value_error(
        self, val_ratio: float, test_ratio: float
    ) -> None:
        """Training always needs a share."""
        with pytest.raises(ValueError, match="ratio"):
            DataModuleConfig(batch_size=8, val_ratio=val_ratio, test_ratio=test_ratio)

    def test__init__when_extra_fields__should_keep_them(self) -> None:
        """Projects add their own loading parameters."""
        config = DataModuleConfig(batch_size=8, extra=self.EXPECTED_EXTRA)

        assert config.model_extra == {"extra": self.EXPECTED_EXTRA}

    @pytest.mark.parametrize(("field", "value"), [("batch_size", 0), ("num_workers", -1)])
    def test__init__when_out_of_range__should_raise_value_error(
        self, field: str, value: float
    ) -> None:
        """Bounds are enforced at validation."""
        params: dict[str, float] = {"batch_size": 8, field: value}

        with pytest.raises(ValueError, match=field):
            DataModuleConfig(**params)


class TestAbstractDataModule:
    """Tests for ``AbstractDataModule`` through the ``RangeDataModule`` implementation."""

    def test__init__should_leave_splits_unset(self, datamodule: RangeDataModule) -> None:
        """Datasets are created in ``setup``, not in ``__init__``."""
        assert datamodule.train_data is None
        assert datamodule.val_data is None
        assert datamodule.test_data is None

    def test__prepare_data__should_run_subclass_hook(self, datamodule: RangeDataModule) -> None:
        """The subclass hook is called."""
        datamodule.prepare_data()

        assert datamodule.prepared is True

    def test__setup__when_fit__should_assign_train_and_val_only(
        self, datamodule: RangeDataModule
    ) -> None:
        """The ``fit`` stage needs train and val."""
        datamodule.setup(stage="fit")

        assert datamodule.train_data is not None
        assert datamodule.val_data is not None
        assert datamodule.test_data is None

    def test__setup__when_test__should_assign_test_only(self, datamodule: RangeDataModule) -> None:
        """The ``test`` stage needs test only."""
        datamodule.setup(stage="test")

        assert datamodule.train_data is None
        assert datamodule.test_data is not None

    def test__setup__when_no_stage__should_assign_every_split(
        self, datamodule: RangeDataModule
    ) -> None:
        """No stage means everything."""
        datamodule.setup()

        assert datamodule.train_data is not None
        assert datamodule.val_data is not None
        assert datamodule.test_data is not None

    def test__train_dataloader__should_shuffle_with_configured_batch_size(
        self, datamodule: RangeDataModule, batch_size: int
    ) -> None:
        """Training shuffles."""
        datamodule.setup(stage="fit")

        loader = datamodule.train_dataloader()

        assert loader.dataset is datamodule.train_data
        assert loader.batch_size == batch_size
        assert isinstance(loader.sampler, RandomSampler)

    @pytest.mark.parametrize("method", ["val_dataloader", "test_dataloader"])
    def test__eval_dataloaders__should_not_shuffle(
        self, datamodule: RangeDataModule, method: str
    ) -> None:
        """Validation and test keep the dataset order."""
        datamodule.setup()

        loader = getattr(datamodule, method)()

        assert isinstance(loader.sampler, SequentialSampler)

    def test__train_dataloader__when_setup_not_called__should_raise_runtime_error(
        self, datamodule: RangeDataModule
    ) -> None:
        """Asking for a loader before ``setup`` is a programming error."""
        with pytest.raises(RuntimeError, match="train_data is not set"):
            datamodule.train_dataloader()

    @pytest.mark.parametrize("method", ["val_dataloader", "test_dataloader"])
    def test__eval_dataloaders__when_no_split__should_return_no_loader(
        self, datamodule: RangeDataModule, method: str
    ) -> None:
        """An empty list is how Lightning is told to skip the phase."""
        datamodule.setup()
        datamodule.val_data = None
        datamodule.test_data = None

        assert getattr(datamodule, method)() == []

    def test__split_indices__should_partition_every_index_by_ratio(
        self, datamodule: RangeDataModule
    ) -> None:
        """The default 70 / 15 / 15 split of 20 samples covers each index exactly once."""
        indices = datamodule.split_indices(20)

        assert [len(indices[phase]) for phase in Phase] == [14, 3, 3]
        assert sorted(index for phase in Phase for index in indices[phase]) == list(range(20))

    def test__split_indices__when_same_seed__should_give_the_same_partition(
        self, datamodule_config: DataModuleConfig
    ) -> None:
        """Splits are reproducible across runs, so phases never leak into each other."""
        first, second = RangeDataModule(datamodule_config), RangeDataModule(datamodule_config)

        assert first.split_indices(20) == second.split_indices(20)

    def test__split_indices__when_a_ratio_is_zero__should_leave_the_phase_out(
        self, datamodule: RangeDataModule
    ) -> None:
        """A task without validation gets no validation indices."""
        datamodule.config.val_ratio, datamodule.config.test_ratio = 0.0, 0.2

        assert set(datamodule.split_indices(10)) == {Phase.TRAIN, Phase.TEST}

    @pytest.mark.filterwarnings("ignore:Length of split at index")
    def test__split_indices__when_a_share_rounds_to_nothing__should_raise_value_error(
        self, datamodule: RangeDataModule
    ) -> None:
        """Two samples cannot feed a 15 % validation and a 15 % test split."""
        with pytest.raises(ValueError, match=r"\['val', 'test'\]"):
            datamodule.split_indices(2)

    def test__val_dataloader__should_yield_batches_of_configured_size(
        self, datamodule: RangeDataModule
    ) -> None:
        """The loader iterates over the data in batches of ``batch_size``."""
        datamodule.setup(stage="fit")

        batches = list(datamodule.val_dataloader())

        assert [batch.tolist() for batch in batches] == [[6, 7], [8]]


class TestSourceDataModule:
    """Tests for ``SourceDataModule`` over twenty records."""

    SAMPLE_COUNT: ClassVar[int] = 20

    @pytest.fixture
    def source(self) -> ListDictSource:
        """Twenty records with one ``value`` each."""
        return ListDictSource([{"value": index} for index in range(self.SAMPLE_COUNT)])

    @pytest.fixture
    def config(self, transform_module: ModuleType) -> SourceDataModuleConfig:
        """``value`` gets ``AddOne`` on training data only."""
        return SourceDataModuleConfig(
            batch_size=2,
            dataset_config=BaseDatasetConfig(
                feature_groups=[
                    FeatureGroupConfig(
                        name="value",
                        source_keys=["value"],
                        pipeline=TransformPipeline(
                            module=transform_module,
                            transforms=[TransformConfig(name="AddOne", apply_on_eval=False)],
                        ),
                    )
                ]
            ),
        )

    @pytest.fixture
    def datamodule(
        self, config: SourceDataModuleConfig, source: ListDictSource
    ) -> SourceDataModule:
        """Data module over the twenty records, not yet set up."""
        return SourceDataModule(config, source)

    def test__setup__should_give_each_phase_a_dataset_of_its_samples(
        self, datamodule: SourceDataModule
    ) -> None:
        """Every record lands in one phase; only the training dataset is training data."""
        datamodule.setup()

        datasets = [datamodule.train_data, datamodule.val_data, datamodule.test_data]
        assert sum(len(dataset) for dataset in datasets if dataset) == self.SAMPLE_COUNT
        assert [dataset.training for dataset in datasets if dataset] == [True, False, False]
