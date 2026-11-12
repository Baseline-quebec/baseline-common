"""Tests for ``BaseDatasetConfig`` and ``BaseDataset``."""

from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest
from pytest_mock import MockerFixture

from baseline_core.core.ml.data.base_dataset import (
    BaseDataset,
    BaseDatasetConfig,
    FeatureGroupConfig,
)
from baseline_core.core.ml.data.cache import DiskTransformCache
from baseline_core.core.ml.data.list_dict_source import ListDictSource
from baseline_core.core.ml.data.transform_pipeline import TransformConfig, TransformPipeline


class TestBaseDatasetConfig:
    """Tests for ``BaseDatasetConfig``."""

    def test__init__when_two_groups_write_the_same_key__should_raise_value_error(
        self, transform_module: ModuleType
    ) -> None:
        """Feature groups cannot overwrite each other's keys."""
        pipeline = TransformPipeline(module=transform_module)

        with pytest.raises(ValueError, match=r"\['feature1'\]"):
            BaseDatasetConfig(
                feature_groups=[
                    FeatureGroupConfig(
                        name="feature1", source_keys=["feature1"], pipeline=pipeline
                    ),
                    FeatureGroupConfig(
                        name="both",
                        source_keys=["feature1", "feature2"],
                        pipeline=pipeline,
                        output_keys=["feature1", "feature2"],
                    ),
                ]
            )


class TestBaseDataset:
    """Tests for ``BaseDataset``."""

    EXPECTED_LENGTH: ClassVar[int] = 3
    EXPECTED_FIRST_SAMPLE: ClassVar[dict[str, object]] = {
        "feature1": 11,
        "feature2": 40,
        "feature3": "hello",
        "total": 30,
    }
    EXPECTED_FIRST_FEATURE1_ON_EVAL: ClassVar[int] = 10
    EXPECTED_SECOND_FEATURE1: ClassVar[int] = 16

    @pytest.fixture
    def scale_both_config(self, transform_module: ModuleType) -> BaseDatasetConfig:
        """``feature1`` and ``feature2`` scaled together, written back as two sample keys."""
        return BaseDatasetConfig(
            feature_groups=[
                FeatureGroupConfig(
                    name="scaled",
                    source_keys=["feature1", "feature2"],
                    pipeline=TransformPipeline(
                        module=transform_module,
                        transforms=[TransformConfig(name="ScaleValues", params={"factor": 3})],
                    ),
                    output_keys=["feature1", "feature2"],
                )
            ]
        )

    @pytest.fixture
    def eval_dataset(
        self, list_source: ListDictSource, dataset_config: BaseDatasetConfig
    ) -> BaseDataset:
        """The dataset as an evaluation split sees it."""
        return BaseDataset(list_source, dataset_config, training=False)

    @pytest.fixture
    def scale_both_dataset(
        self, list_source: ListDictSource, scale_both_config: BaseDatasetConfig
    ) -> BaseDataset:
        """Dataset whose one group writes two sample keys."""
        return BaseDataset(list_source, scale_both_config)

    @pytest.fixture
    def ghost_output_dataset(
        self, list_source: ListDictSource, scale_both_config: BaseDatasetConfig
    ) -> BaseDataset:
        """Dataset whose group declares an output key its pipeline never returns."""
        scale_both_config.feature_groups[0].output_keys = ["feature1", "ghost"]
        return BaseDataset(list_source, scale_both_config)

    @pytest.fixture
    def summing_dataset(
        self, list_source: ListDictSource, transform_module: ModuleType
    ) -> BaseDataset:
        """Dataset whose group declares output keys but returns a single number."""
        config = BaseDatasetConfig(
            feature_groups=[
                FeatureGroupConfig(
                    name="total",
                    source_keys=["feature1", "feature2"],
                    pipeline=TransformPipeline(
                        module=transform_module, transforms=[TransformConfig(name="SumValues")]
                    ),
                    output_keys=["total"],
                )
            ]
        )
        return BaseDataset(list_source, config)

    def test__init__when_source_key_missing__should_raise_value_error(
        self, list_source: ListDictSource, transform_module: ModuleType
    ) -> None:
        """Feature groups are validated against the source fields up front."""
        config = BaseDatasetConfig(
            feature_groups=[
                FeatureGroupConfig(
                    name="ghost",
                    source_keys=["missing"],
                    pipeline=TransformPipeline(module=transform_module),
                )
            ]
        )

        with pytest.raises(ValueError, match=r"Source keys \['missing'\] of feature group"):
            BaseDataset(list_source, config)

    def test__len__should_return_source_length(self, base_dataset: BaseDataset) -> None:
        """Length is the source length."""
        assert len(base_dataset) == self.EXPECTED_LENGTH

    def test__getitem__should_build_every_feature_group(self, base_dataset: BaseDataset) -> None:
        """Each group runs its pipeline; multi-key groups receive a dict."""
        sample = base_dataset[0]

        assert sample == self.EXPECTED_FIRST_SAMPLE

    def test__getitem__when_other_index__should_use_that_sample(
        self, base_dataset: BaseDataset
    ) -> None:
        """The index selects the source sample."""
        sample = base_dataset[1]

        assert sample["feature1"] == self.EXPECTED_SECOND_FEATURE1
        assert sample["feature3"] == "world"

    def test__getitem__when_not_training__should_skip_training_only_steps(
        self, eval_dataset: BaseDataset
    ) -> None:
        """An evaluation split does not run steps with ``apply_on_eval=False``."""
        sample = eval_dataset[0]

        assert sample["feature1"] == self.EXPECTED_FIRST_FEATURE1_ON_EVAL
        assert sample["feature2"] == self.EXPECTED_FIRST_SAMPLE["feature2"]

    def test__getitem__when_output_keys__should_spread_the_pipeline_result(
        self, scale_both_dataset: BaseDataset
    ) -> None:
        """Features transformed together come out as separate sample keys."""
        sample = scale_both_dataset[0]

        assert sample == {"feature1": 30, "feature2": 60}

    def test__getitem__when_output_key_missing__should_raise_key_error(
        self, ghost_output_dataset: BaseDataset
    ) -> None:
        """A declared output key the pipeline does not return is named."""
        with pytest.raises(KeyError, match="ghost"):
            ghost_output_dataset[0]

    def test__getitem__when_output_keys_but_not_a_mapping__should_raise_type_error(
        self, summing_dataset: BaseDataset
    ) -> None:
        """Output keys need the pipeline to return a mapping."""
        with pytest.raises(TypeError, match="returned int"):
            summing_dataset[0]

    def test__select__should_keep_the_groups_over_fewer_samples(
        self, base_dataset: BaseDataset
    ) -> None:
        """A phase's view has the chosen samples, the same feature groups and its own flag."""
        selected = base_dataset.select([1], training=False)

        assert len(selected) == 1
        assert selected.training is False
        assert selected[0]["feature3"] == "world"
        assert selected.config is base_dataset.config


class TestBaseDatasetCache:
    """Tests for a ``BaseDataset`` reading its cached steps from a ``DiskTransformCache``."""

    EXPECTED_TRAINING_FEATURE1: ClassVar[int] = 21
    EXPECTED_EVAL_FEATURE1: ClassVar[int] = 20
    EXPECTED_TRIPLED_FEATURE1: ClassVar[int] = 31

    @staticmethod
    def multiplying_config(transform_module: ModuleType, factor: int) -> BaseDatasetConfig:
        """``feature1 * factor`` cached, then ``+ 1`` on the fly and on training only."""
        return BaseDatasetConfig(
            feature_groups=[
                FeatureGroupConfig(
                    name="feature1",
                    source_keys=["feature1"],
                    pipeline=TransformPipeline(
                        module=transform_module,
                        transforms=[
                            TransformConfig(
                                name="Multiply", params={"factor": factor}, apply_on_the_fly=False
                            ),
                            TransformConfig(name="AddOne", apply_on_eval=False),
                        ],
                    ),
                )
            ]
        )

    @pytest.fixture
    def cache(self, tmp_path: Path) -> DiskTransformCache:
        """Cache in a temporary folder."""
        return DiskTransformCache(tmp_path / "cache")

    @pytest.fixture
    def cached_dataset(
        self, list_source: ListDictSource, transform_module: ModuleType, cache: DiskTransformCache
    ) -> BaseDataset:
        """Training dataset whose ``feature1`` is doubled once, then incremented on every read."""
        return BaseDataset(list_source, self.multiplying_config(transform_module, 2), cache=cache)

    @pytest.fixture
    def tripling_dataset(
        self, list_source: ListDictSource, transform_module: ModuleType, cache: DiskTransformCache
    ) -> BaseDataset:
        """The same dataset with ``Multiply(3)`` cached, sharing the cache folder."""
        return BaseDataset(list_source, self.multiplying_config(transform_module, 3), cache=cache)

    def test__getitem__when_read_twice__should_prepare_the_sample_once(
        self, cached_dataset: BaseDataset, mocker: MockerFixture
    ) -> None:
        """The second read comes from the cache and still runs the live step."""
        prepare = mocker.spy(TransformPipeline, "prepare")

        first = cached_dataset[0]["feature1"]
        second = cached_dataset[0]["feature1"]

        assert first == second == self.EXPECTED_TRAINING_FEATURE1
        assert prepare.call_count == 1

    def test__select__when_not_training__should_reuse_the_training_cache(
        self, cached_dataset: BaseDataset, mocker: MockerFixture
    ) -> None:
        """Every split reads the same cached result and runs its own live steps."""
        cached_dataset[0]
        prepare = mocker.spy(TransformPipeline, "prepare")

        evaluated = cached_dataset.select([0], training=False)[0]["feature1"]

        assert evaluated == self.EXPECTED_EVAL_FEATURE1
        assert prepare.call_count == 0

    def test__getitem__when_cached_step_changes__should_not_reuse_the_old_result(
        self, cached_dataset: BaseDataset, tripling_dataset: BaseDataset
    ) -> None:
        """A changed cached step gets its own key in the shared folder."""
        cached_dataset[0]

        tripled = tripling_dataset[0]["feature1"]

        assert tripled == self.EXPECTED_TRIPLED_FEATURE1
