"""Tests for the head configuration and the heads."""

import pytest
import torch
from torch import nn

from baseline_core.core.ml.data.batch import Batch, TensorTargets
from baseline_core.core.ml.model.abstract_head import HeadConfig
from tests.core.ml.dummy_classes import DummyBackbone, DummyHead


class TestHeadConfig:
    """Tests for ``HeadConfig``."""

    def test__init__when_extra_fields__should_keep_them(self) -> None:
        """Projects extend heads with their own fields."""
        config = HeadConfig(target="y", hidden_dim=8)

        assert config.target == "y"
        assert config.weight == pytest.approx(1.0)
        assert config.model_extra == {"hidden_dim": 8}

    def test__init__when_weight_not_positive__should_raise_value_error(self) -> None:
        """A head with no weight would not train."""
        with pytest.raises(ValueError, match="greater than 0"):
            HeadConfig(target="y", weight=0.0)


class TestAbstractHead:
    """Tests for an ``AbstractHead`` implementation."""

    def test__forward__should_predict_from_its_group(
        self,
        backbone: DummyBackbone,
        batch: Batch[TensorTargets],
        mse_head: DummyHead,
        user_count: int,
    ) -> None:
        """A head reads the embeddings of its group and returns one prediction per sample."""
        features = backbone(batch)

        prediction = mse_head(features, batch)

        assert prediction.shape == (user_count, 1)

    def test__loss__should_score_predictions_against_its_target(
        self, backbone: DummyBackbone, batch: Batch[TensorTargets], mse_head: DummyHead
    ) -> None:
        """The head reads its own supervision in the batch."""
        prediction = mse_head(backbone(batch), batch)
        assert batch.targets is not None

        loss = mse_head.loss(prediction, batch)

        assert isinstance(loss, torch.Tensor)
        assert torch.equal(loss, nn.functional.mse_loss(prediction, batch.targets["y_user"]))
