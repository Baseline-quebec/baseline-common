"""Fixtures for the dataset layer: raw records, sources, feature groups and a dataset."""

from types import ModuleType
from typing import Any

import pytest

from baseline_core.core.ml.data.base_dataset import (
    BaseDataset,
    BaseDatasetConfig,
    FeatureGroupConfig,
)
from baseline_core.core.ml.data.list_dict_source import ListDictSource
from baseline_core.core.ml.data.taxonomy import LabelDefinition, Taxonomy
from baseline_core.core.ml.data.transform_pipeline import TransformConfig, TransformPipeline
from tests.core.ml.dummy_classes import UpperCaseLoader


@pytest.fixture
def records() -> list[dict[str, Any]]:
    """Three raw samples with two numeric keys and one text key."""
    return [
        {"feature1": 10, "feature2": 20, "feature3": "hello"},
        {"feature1": 15, "feature2": 25, "feature3": "world"},
        {"feature1": 20, "feature2": 30, "feature3": "test"},
    ]


@pytest.fixture
def list_source(records: list[dict[str, Any]]) -> ListDictSource:
    """Source over the raw records."""
    return ListDictSource(records)


@pytest.fixture
def loaded_source(records: list[dict[str, Any]]) -> ListDictSource:
    """Source over the raw records whose ``feature3`` is loaded when read."""
    return ListDictSource(records, loaders={"feature3": UpperCaseLoader()})


@pytest.fixture
def empty_source() -> ListDictSource:
    """Source without any record."""
    return ListDictSource([])


@pytest.fixture
def feature_groups(transform_module: ModuleType) -> list[FeatureGroupConfig]:
    """``feature1 + 1`` (training), ``feature2 * 2``, ``str(feature3)``, their sum."""

    def pipeline(
        name: str, params: dict[str, Any] | None = None, *, apply_on_eval: bool = True
    ) -> TransformPipeline:
        """One-step pipeline of the transform ``name``."""
        return TransformPipeline(
            module=transform_module,
            transforms=[
                TransformConfig(name=name, params=params or {}, apply_on_eval=apply_on_eval)
            ],
        )

    return [
        FeatureGroupConfig(
            name="feature1",
            source_keys=["feature1"],
            pipeline=pipeline("AddOne", apply_on_eval=False),
        ),
        FeatureGroupConfig(
            name="feature2", source_keys=["feature2"], pipeline=pipeline("Multiply", {"factor": 2})
        ),
        FeatureGroupConfig(name="feature3", source_keys=["feature3"], pipeline=pipeline("ToText")),
        FeatureGroupConfig(
            name="total", source_keys=["feature1", "feature2"], pipeline=pipeline("SumValues")
        ),
    ]


@pytest.fixture
def dataset_config(feature_groups: list[FeatureGroupConfig]) -> BaseDatasetConfig:
    """Every feature group."""
    return BaseDatasetConfig(feature_groups=feature_groups)


@pytest.fixture
def base_dataset(list_source: ListDictSource, dataset_config: BaseDatasetConfig) -> BaseDataset:
    """Training dataset over the records with every feature group."""
    return BaseDataset(list_source, dataset_config)


@pytest.fixture
def taxonomy() -> Taxonomy:
    """Three labels, one of them still a candidate."""
    return Taxonomy(
        labels=[
            LabelDefinition(name="box", description="A box.", status="retained"),
            LabelDefinition(name="bag", description="A bag.", status="retained"),
            LabelDefinition(name="sauce", description="A sauce cup.", status="candidate"),
        ]
    )
