"""Tests for the container running every head."""

from typing import Any

import pytest
import torch
from torch import nn

from baseline_core.core.ml.data.batch import Batch, TensorTargets
from baseline_core.core.ml.model.head_container import HeadContainer
from tests.core.ml.dummy_classes import DummyBackbone, DummyHead, DummyHeadConfig, LoggingHead


class TestHeadContainer:
    """Tests for ``HeadContainer``."""

    @pytest.fixture
    def predictions(
        self, backbone: DummyBackbone, heads: HeadContainer, batch: Batch[TensorTargets]
    ) -> dict[str, Any]:
        """Predictions of both heads on ``batch``."""
        return heads(backbone(batch), batch)

    def test__forward__should_gather_every_head_prediction(
        self, predictions: dict[str, Any], flight_count: int
    ) -> None:
        """Predictions are keyed by target."""
        assert set(predictions) == {"y_user", "y_flight"}
        assert predictions["y_flight"].shape == (flight_count, 1)

    def test__loss__should_weight_every_head_loss_into_the_total(
        self,
        heads: HeadContainer,
        predictions: dict[str, Any],
        batch: Batch[TensorTargets],
        mse_head_config: DummyHeadConfig,
        bce_head_config: DummyHeadConfig,
    ) -> None:
        """Each head's loss is reported under ``<target>_loss`` and weighted into ``loss``."""
        losses = heads.loss(predictions, batch)

        assert set(losses) == {"loss", "y_user_loss", "y_flight_loss"}
        expected = (
            mse_head_config.weight * losses["y_user_loss"]
            + bce_head_config.weight * losses["y_flight_loss"]
        )
        assert torch.allclose(losses["loss"], expected)

    @pytest.fixture
    def logging_heads(self, mse_head_config: DummyHeadConfig) -> HeadContainer:
        """One head reporting the mean of its predictions with its loss."""
        return HeadContainer([LoggingHead(mse_head_config, nn.MSELoss())])

    @pytest.fixture
    def unreduced_heads(self, mse_head_config: DummyHeadConfig) -> HeadContainer:
        """One head whose loss keeps one value per element."""
        return HeadContainer([DummyHead(mse_head_config, nn.MSELoss(reduction="none"))])

    def test__init__when_targets_repeat__should_raise_value_error(
        self, mse_head: DummyHead, mse_head_config: DummyHeadConfig
    ) -> None:
        """Predictions, losses and metrics are keyed by target, so targets are unique."""
        twin = DummyHead(mse_head_config, nn.MSELoss())

        with pytest.raises(ValueError, match=r"duplicates: \['y_user'\]"):
            HeadContainer([mse_head, twin])

    def test__loss__when_loss_result__should_report_its_extra_values_under_the_target(
        self, logging_heads: HeadContainer, backbone: DummyBackbone, batch: Batch[TensorTargets]
    ) -> None:
        """Extra fields of a ``LossResult`` are reported, detached, as ``<target>_<field>``."""
        predictions = logging_heads(backbone(batch), batch)

        losses = logging_heads.loss(predictions, batch)

        assert set(losses) == {"loss", "y_user_loss", "y_user_pred_mean"}
        assert torch.equal(losses["y_user_pred_mean"], predictions["y_user"].mean())
        assert not losses["y_user_pred_mean"].requires_grad

    def test__loss__when_not_scalar__should_raise_value_error(
        self, unreduced_heads: HeadContainer, backbone: DummyBackbone, batch: Batch[TensorTargets]
    ) -> None:
        """A loss that is not reduced to a scalar is a head bug, not something to average."""
        predictions = unreduced_heads(backbone(batch), batch)

        with pytest.raises(ValueError, match=r"not scalar: \['y_user_loss'\]"):
            unreduced_heads.loss(predictions, batch)
