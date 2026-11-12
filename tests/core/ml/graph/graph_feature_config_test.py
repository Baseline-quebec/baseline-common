"""Tests for the graph feature config."""

import pytest
from pydantic import ValidationError

from baseline_core.core.ml.graph.graph_feature_config import Feature, GraphFeatureConfig


class TestGraphFeatureConfig:
    def test__features__load_from_plain_data(self) -> None:
        """A config section of names per node type loads as features, without transforms."""
        config = GraphFeatureConfig.model_validate({"features": {"product": [{"name": "price"}]}})

        assert config.features == {"product": [Feature(name="price")]}
        assert config.features["product"][0].transform is None

    def test__a_feature_without_a_name__is_refused(self) -> None:
        """Every feature names the node attribute it reads."""
        with pytest.raises(ValidationError, match="name"):
            GraphFeatureConfig.model_validate({"features": {"product": [{}]}})

    def test__a_feature__is_frozen(self) -> None:
        """A feature read by a converter cannot change under it."""
        feature = Feature(name="price")

        with pytest.raises(ValidationError, match="frozen"):
            feature.name = "weight"  # type: ignore[misc]
