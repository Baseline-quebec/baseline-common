"""Tests for transform pipelines."""

from types import ModuleType
from typing import ClassVar

import pytest
import torch

from baseline_core.core.ml.data.transform_pipeline import TransformConfig, TransformPipeline


class TestTransformPipeline:
    """Tests for ``TransformPipeline``."""

    INPUT: ClassVar[int] = 5
    EXPECTED_ADD_THEN_MULTIPLY: ClassVar[int] = 12
    EXPECTED_STEP_COUNT: ClassVar[int] = 2

    @pytest.fixture
    def pipeline(self, transform_module: ModuleType) -> TransformPipeline:
        """``AddOne`` (training only) followed by ``Multiply(2)``."""
        return TransformPipeline(
            module=transform_module,
            transforms=[
                TransformConfig(name="AddOne", apply_on_eval=False),
                TransformConfig(name="Multiply", params={"factor": 2}),
            ],
        )

    def test__build_steps__when_validated__should_instantiate_every_transform(
        self, pipeline: TransformPipeline
    ) -> None:
        """Steps are built once, at validation time."""
        steps = pipeline._steps

        assert len(steps) == self.EXPECTED_STEP_COUNT

    def test__call__when_training__should_apply_steps_in_order(
        self, pipeline: TransformPipeline
    ) -> None:
        """Every step runs, in declaration order."""
        result = pipeline(self.INPUT)

        assert result == self.EXPECTED_ADD_THEN_MULTIPLY

    def test__call__when_not_training__should_skip_train_only_steps(
        self, pipeline: TransformPipeline
    ) -> None:
        """Steps with ``apply_on_eval=False`` are skipped outside training."""
        result = pipeline(self.INPUT, training=False)

        assert result == self.INPUT * 2

    def test__init__when_module_is_dotted_path__should_import_it(self) -> None:
        """The module can be given as a string."""
        pipeline = TransformPipeline(
            module="torch.nn", transforms=[TransformConfig(name="Identity")]
        )

        assert pipeline(self.INPUT) == self.INPUT

    def test__init__when_class_missing__should_raise_import_error(
        self, transform_module: ModuleType
    ) -> None:
        """An unknown transform name fails at validation, not on first use."""
        with pytest.raises(ImportError, match="'Missing' not found"):
            TransformPipeline(
                module=transform_module, transforms=[TransformConfig(name="Missing")]
            )

    def test__init__when_dtype_given_by_name__should_resolve_it(self) -> None:
        """A dtype written in YAML reaches the constructor as a ``torch.dtype``."""
        pipeline = TransformPipeline(
            module="torchvision.transforms.v2",
            transforms=[TransformConfig(name="ToDtype", params={"dtype": "float32"})],
        )

        result = pipeline(torch.zeros(2, dtype=torch.uint8))

        assert result.dtype == torch.float32

    def test__init__when_dtype_name_unknown__should_raise_value_error(self) -> None:
        """A misspelled dtype fails at validation."""
        with pytest.raises(ValueError, match="'float33' is not a torch dtype"):
            TransformPipeline(
                module="torchvision.transforms.v2",
                transforms=[TransformConfig(name="ToDtype", params={"dtype": "float33"})],
            )

    def test__init__when_nested_transforms__should_build_them_into_the_enclosing_step(
        self,
    ) -> None:
        """``RandomApply`` receives its steps built from the same module."""
        pipeline = TransformPipeline(
            module="torchvision.transforms.v2",
            transforms=[
                TransformConfig(
                    name="RandomApply",
                    params={"p": 1.0},
                    transforms=[TransformConfig(name="ToDtype", params={"dtype": "float64"})],
                )
            ],
        )

        result = pipeline(torch.zeros(2, dtype=torch.uint8))

        assert result.dtype == torch.float64

    def test__model_dump__should_exclude_module(self, pipeline: TransformPipeline) -> None:
        """The module object is not part of the serialized config."""
        dumped = pipeline.model_dump()

        assert "module" not in dumped
        assert dumped["transforms"][0]["apply_on_the_fly"] is True


class TestTransformPipelineCachedSteps:
    """Tests for the steps a ``TransformPipeline`` caches (``apply_on_the_fly=False``)."""

    INPUT: ClassVar[int] = 5
    EXPECTED_PREPARED: ClassVar[int] = 10
    EXPECTED_TRANSFORMED: ClassVar[int] = 11

    @pytest.fixture
    def pipeline(self, transform_module: ModuleType) -> TransformPipeline:
        """``Multiply(2)`` cached, then ``AddOne`` run on the fly and on training only."""
        return TransformPipeline(
            module=transform_module,
            transforms=[
                TransformConfig(name="Multiply", params={"factor": 2}, apply_on_the_fly=False),
                TransformConfig(name="AddOne", apply_on_eval=False),
            ],
        )

    @pytest.fixture
    def tripling_pipeline(self, transform_module: ModuleType) -> TransformPipeline:
        """The same pipeline with ``Multiply(3)`` cached instead."""
        return TransformPipeline(
            module=transform_module,
            transforms=[
                TransformConfig(name="Multiply", params={"factor": 3}, apply_on_the_fly=False),
                TransformConfig(name="AddOne", apply_on_eval=False),
            ],
        )

    @pytest.fixture
    def live_pipeline(self, transform_module: ModuleType) -> TransformPipeline:
        """``Multiply(2)`` run on the fly: nothing to cache."""
        return TransformPipeline(
            module=transform_module,
            transforms=[TransformConfig(name="Multiply", params={"factor": 2})],
        )

    def test__prepare__should_run_only_the_cached_steps(self, pipeline: TransformPipeline) -> None:
        """The live steps are left for every read."""
        assert pipeline.prepare(self.INPUT) == self.EXPECTED_PREPARED

    def test__call__when_prepared__should_run_only_the_live_steps(
        self, pipeline: TransformPipeline
    ) -> None:
        """A prepared value does not go through the cached steps twice."""
        assert pipeline(self.EXPECTED_PREPARED, prepared=True) == self.EXPECTED_TRANSFORMED

    def test__call__when_not_prepared__should_run_every_step(
        self, pipeline: TransformPipeline
    ) -> None:
        """Without a cache, the pipeline behaves as if no step were cached."""
        assert pipeline(self.INPUT) == self.EXPECTED_TRANSFORMED

    def test__cache_key__should_change_with_the_cached_steps(
        self, pipeline: TransformPipeline, tripling_pipeline: TransformPipeline
    ) -> None:
        """Another cached step means another key, so old results are never reused."""
        assert pipeline.cache_key is not None
        assert pipeline.cache_key != tripling_pipeline.cache_key

    def test__cache_key__when_every_step_is_live__should_be_none(
        self, live_pipeline: TransformPipeline
    ) -> None:
        """A pipeline without cached steps has nothing to cache."""
        assert live_pipeline.cache_key is None

    def test__init__when_cached_step_follows_a_live_one__should_raise_value_error(
        self, transform_module: ModuleType
    ) -> None:
        """A cached result cannot depend on a step that runs on every read."""
        with pytest.raises(ValueError, match="follows the live step 'AddOne'"):
            TransformPipeline(
                module=transform_module,
                transforms=[
                    TransformConfig(name="AddOne"),
                    TransformConfig(name="Multiply", params={"factor": 2}, apply_on_the_fly=False),
                ],
            )

    def test__init__when_cached_step_skipped_on_eval__should_raise_value_error(
        self, transform_module: ModuleType
    ) -> None:
        """A cached result is shared by every split, so it cannot be training-only."""
        with pytest.raises(ValueError, match="cached but skipped outside training"):
            TransformPipeline(
                module=transform_module,
                transforms=[
                    TransformConfig(name="AddOne", apply_on_the_fly=False, apply_on_eval=False)
                ],
            )


class TestTransformConfig:
    """Tests for ``TransformConfig`` defaults."""

    def test__init__when_defaults__should_apply_everywhere(self) -> None:
        """A step applies on eval and on the fly unless said otherwise."""
        config = TransformConfig(name="AddOne")

        assert config.apply_on_eval is True
        assert config.apply_on_the_fly is True
        assert config.params == {}
