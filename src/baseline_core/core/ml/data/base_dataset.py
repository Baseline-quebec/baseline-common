"""Map-style dataset whose samples are feature groups built from a data source."""

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator
from torch.utils.data import Dataset

from baseline_core.core.ml.data.abstract_data_source import AbstractDataSource
from baseline_core.core.ml.data.cache import AbstractTransformCache
from baseline_core.core.ml.data.transform_pipeline import TransformPipeline


class FeatureGroupConfig(BaseModel):
    """Features read together from the source, transformed by one pipeline.

    Attributes:
        name: Sample key of the result (e.g. ``"image"``, ``"label"``).
        source_keys: Features read from the source; one is passed as is, several as a ``dict``.
        pipeline: Transforms turning the loaded feature(s) into the result.
        output_keys: Sample keys taken from the ``dict`` the pipeline returns, instead of ``name``.
    """

    name: str
    source_keys: list[str] = Field(min_length=1)
    pipeline: TransformPipeline
    output_keys: list[str] = Field(default_factory=list)

    @property
    def sample_keys(self) -> list[str]:
        """Keys this group writes in a sample."""
        return self.output_keys or [self.name]


class BaseDatasetConfig(BaseModel):
    """The feature groups a ``BaseDataset`` builds for every sample.

    Attributes:
        feature_groups: Feature groups, each writing one or more keys of a sample.
    """

    feature_groups: list[FeatureGroupConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_sample_keys(self) -> Self:
        """Two groups never write the same sample key."""
        keys = [key for group in self.feature_groups for key in group.sample_keys]
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)

        if duplicates:
            error_msg = f"Sample keys written by more than one feature group: {duplicates}"
            raise ValueError(error_msg)

        return self


class BaseDataset(Dataset[dict[str, Any]]):
    """Dataset mapping ``source sample -> feature group pipelines -> dict[str, Any]``.

    Attributes:
        source: Where the features come from, loaded.
        config: Feature groups built for every sample.
        training: Whether this dataset is training data.
        cache: Results of the cached transform steps, or ``None`` to run every step on every read.
    """

    def __init__(
        self,
        source: AbstractDataSource[Any],
        config: BaseDatasetConfig,
        *,
        training: bool = True,
        cache: AbstractTransformCache | None = None,
    ) -> None:
        """Validate that every source key exists, then keep the configuration.

        Args:
            source: Data source.
            config: Feature groups to build.
            training: Whether training-only steps (``apply_on_eval=False``) run.
            cache: Results of the cached transform steps, or ``None``.

        Raises:
            ValueError: If a feature group reads a feature missing from ``source``.
        """
        super().__init__()
        self.source = source
        self.config = config
        self.training = training
        self.cache = cache

        available = set(source.field_names)

        for group in config.feature_groups:
            missing = [key for key in group.source_keys if key not in available]

            if missing:
                error_msg = (
                    f"Source keys {missing} of feature group '{group.name}' not found in data "
                    f"source. Available keys: {sorted(available)}"
                )
                raise ValueError(error_msg)

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.source)

    def select(self, indices: Sequence[int], *, training: bool) -> "BaseDataset":
        """Return a dataset over the samples at ``indices`` only, e.g. one phase's.

        Args:
            indices: Sample indices.
            training: Whether the returned dataset is training data.

        Returns:
            A dataset with the same feature groups over the selected samples.
        """
        return BaseDataset(
            self.source.select(indices), self.config, training=training, cache=self.cache
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Build every feature group of sample ``index``.

        Raises:
            TypeError: If a group with ``output_keys`` gets something else than a mapping.
            KeyError: If that mapping lacks one of its ``output_keys``.
        """
        sample = {}

        for group in self.config.feature_groups:
            value = self.build_group(index, group)

            if not group.output_keys:
                sample[group.name] = value
                continue

            if not isinstance(value, Mapping):
                error_msg = (
                    f"Feature group '{group.name}' declares output keys {group.output_keys} "
                    f"but its pipeline returned {type(value).__name__}, not a mapping."
                )
                raise TypeError(error_msg)

            missing = [key for key in group.output_keys if key not in value]

            if missing:
                error_msg = f"Pipeline of feature group '{group.name}' returned no {missing}."
                raise KeyError(error_msg)

            sample.update({key: value[key] for key in group.output_keys})

        return sample

    def build_group(self, index: int, group: FeatureGroupConfig) -> Any:  # noqa: ANN401
        """Run the pipeline of ``group`` on sample ``index``, its cached steps read from the cache.

        Args:
            index: Sample index.
            group: Feature group to build.

        Returns:
            The pipeline output.
        """
        pipeline_key = group.pipeline.cache_key

        if self.cache is None or pipeline_key is None:
            return group.pipeline(self.read_group(index, group), training=self.training)

        key = self.sample_cache_key(index, group, pipeline_key)
        prepared = self.cache.load(key)

        if prepared is None:
            prepared = group.pipeline.prepare(self.read_group(index, group))
            self.cache.save(key, prepared)

        return group.pipeline(prepared, training=self.training, prepared=True)

    def read_group(self, index: int, group: FeatureGroupConfig) -> Any:  # noqa: ANN401
        """The loaded feature of ``group``, or a dict of its features when it reads several.

        Args:
            index: Sample index.
            group: Feature group to read.

        Returns:
            What the pipeline of ``group`` starts from.
        """
        if len(group.source_keys) == 1:
            return self.source.get_value(index, group.source_keys[0])

        return {key: self.source.get_value(index, key) for key in group.source_keys}

    def sample_cache_key(self, index: int, group: FeatureGroupConfig, pipeline_key: str) -> str:
        """Cache key of ``group`` for sample ``index``, from its cached steps and stored values.

        Args:
            index: Sample index.
            group: Feature group.
            pipeline_key: ``cache_key`` of the group's pipeline.

        Returns:
            ``<group name>-<hash>``, shared by every split; a moved file gets another key.
        """
        stored = [repr(self.source.get_stored_value(index, key)) for key in group.source_keys]
        digest = hashlib.sha256("\x1f".join([pipeline_key, *stored]).encode()).hexdigest()
        return f"{group.name}-{digest[:24]}"
