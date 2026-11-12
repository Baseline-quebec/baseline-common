"""Tests for the Hugging Face heads and their configs."""

from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture
from torchmetrics import MeanSquaredError, MetricCollection

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.data.phase import Phase
from baseline_core.core.ml.huggingface.hf_head import HFHead, HFHeadConfig
from baseline_core.core.ml.huggingface.passthrough_backbone import PassthroughBackbone
from baseline_core.core.ml.model.head_container import HeadContainer
from baseline_core.core.ml.training.backbone_heads_module import BackboneHeadsModule
from baseline_core.core.ml.training.optimizer import OptimizerFactory


class TestHFHeadConfig:
    """Tests for the validation of ``HFHeadConfig``."""

    @pytest.mark.parametrize("field", ["auto_class", "processor_class"])
    def test__init__when_class_unknown__should_raise_value_error(self, field: str) -> None:
        """Class names must exist in ``transformers``."""
        fields = {
            "target": "y",
            "model_id": "dummy",
            "auto_class": "AutoModel",
            field: "AutoGhost",
        }

        with pytest.raises(ValueError, match="transformers has no class 'AutoGhost'"):
            HFHeadConfig.model_validate(fields)


class TestHFHead:
    """Tests for ``HFHead`` loaded from a local folder holding a dummy model."""

    @pytest.fixture
    def features(self, regression_batch: Batch[Any]) -> dict[str, torch.Tensor]:
        """What ``PassthroughBackbone`` hands the head: the batch inputs."""
        return PassthroughBackbone()(regression_batch)

    @pytest.fixture
    def frozen_config(self, hf_head_config: HFHeadConfig) -> HFHeadConfig:
        """The dummy model with its encoder frozen."""
        return hf_head_config.model_copy(update={"freeze": ["encoder"]})

    @pytest.fixture
    def frozen_head(self, frozen_config: HFHeadConfig) -> HFHead:
        """The dummy model with its encoder frozen, as a head."""
        return HFHead(frozen_config)

    @pytest.fixture
    def frozen_module(
        self, frozen_head: HFHead, optimizer_factory: OptimizerFactory
    ) -> BackboneHeadsModule:
        """The frozen head alone in a module."""
        return BackboneHeadsModule(
            PassthroughBackbone(), HeadContainer([frozen_head]), optimizer_factory
        )

    @pytest.fixture
    def measured_module(
        self, hf_head: HFHead, hf_head_config: HFHeadConfig, optimizer_factory: OptimizerFactory
    ) -> BackboneHeadsModule:
        """The head alone in a module measuring its mean squared error."""
        return BackboneHeadsModule(
            PassthroughBackbone(),
            HeadContainer([hf_head]),
            optimizer_factory,
            {hf_head_config.target: MetricCollection([MeanSquaredError()])},
        )

    def test__init__should_keep_the_mode_from_pretrained_gives(self, hf_head: HFHead) -> None:
        """Construction forces no mode: a loaded model stays in evaluation mode."""
        assert not hf_head.model.training

    def test__on_fit_start__should_put_the_model_in_training_mode(
        self, hf_module: BackboneHeadsModule, hf_head: HFHead
    ) -> None:
        """Training starts in training mode, which neither the loader nor Lightning sets."""
        hf_module.on_fit_start()

        assert all(module.training for module in hf_head.model.modules())

    def test__loss__should_return_the_loss_computed_by_the_model(
        self, hf_head: HFHead, regression_batch: Batch[Any], features: dict[str, torch.Tensor]
    ) -> None:
        """The model computes its own loss from ``labels``."""
        expected = hf_head.model(**regression_batch.x, labels=regression_batch.targets).loss

        result = hf_head.loss(hf_head(features, regression_batch), regression_batch)

        assert torch.equal(result.loss, expected)
        assert result.model_extra == {}

    def test__loss__when_no_targets__should_raise_value_error(
        self, hf_head: HFHead, regression_batch: Batch[Any], features: dict[str, torch.Tensor]
    ) -> None:
        """Training steps need supervision."""
        batch = regression_batch.model_copy(update={"targets": None})
        predictions = hf_head(features, batch)

        with pytest.raises(ValueError, match="without targets"):
            hf_head.loss(predictions, batch)

    def test__postprocess__should_return_outputs_unchanged(
        self, hf_head: HFHead, regression_batch: Batch[Any], features: dict[str, torch.Tensor]
    ) -> None:
        """Task heads override ``postprocess``; the base returns the model outputs."""
        outputs = hf_head(features, regression_batch)

        assert hf_head.postprocess(outputs, regression_batch) is outputs

    def test__update_metrics__should_compare_postprocessed_outputs_with_targets(
        self,
        measured_module: BackboneHeadsModule,
        hf_head: HFHead,
        hf_head_config: HFHeadConfig,
        regression_batch: Batch[Any],
        mocker: MockerFixture,
    ) -> None:
        """Metrics see what ``postprocess`` and ``postprocess_targets`` return."""
        output = measured_module.step(regression_batch)
        mocker.patch.object(hf_head, "postprocess", return_value=torch.ones(3, 1))

        measured_module.update_metrics(Phase.VAL, output, regression_batch)

        computed = measured_module.metrics_for(Phase.VAL)[hf_head_config.target].compute()
        assert computed["val_regression_MeanSquaredError"].item() == pytest.approx(1.0)

    def test__predict_step__should_run_the_model_without_targets(
        self,
        hf_module: BackboneHeadsModule,
        hf_head: HFHead,
        hf_head_config: HFHeadConfig,
        regression_batch: Batch[Any],
    ) -> None:
        """Prediction runs the model on the inputs alone."""
        inference_batch = regression_batch.model_copy(update={"targets": None})
        hf_module.eval()

        predictions = hf_module.predict_step(inference_batch)[hf_head_config.target]

        assert torch.equal(predictions.logits, hf_head.model(**inference_batch.x).logits)
        assert predictions.loss is None

    def test__init__when_freeze_prefix__should_freeze_matching_parameters(
        self, frozen_head: HFHead, frozen_module: BackboneHeadsModule
    ) -> None:
        """Frozen parameters stay out of the optimizer."""
        optimized = frozen_module.configure_optimizers()["optimizer"].param_groups[0]["params"]

        trainable = {
            name for name, param in frozen_head.model.named_parameters() if param.requires_grad
        }
        assert trainable == {"head.weight", "head.bias"}
        assert len(optimized) == len(trainable)

    def test__init__when_freeze_prefix_matches_nothing__should_raise_value_error(
        self, hf_head_config: HFHeadConfig
    ) -> None:
        """A typo in a freeze prefix must not silently train everything."""
        config = hf_head_config.model_copy(update={"freeze": ["backbone"]})

        with pytest.raises(ValueError, match=r"match no parameter: \['backbone'\]"):
            HFHead(config)

    def test__train__when_freeze_prefix__should_keep_frozen_submodules_in_eval_mode(
        self, frozen_head: HFHead
    ) -> None:
        """Frozen layers keep their dropout off and their batch norm statistics fixed."""
        frozen_head.train()

        assert not frozen_head.model.get_submodule("encoder").training
        assert frozen_head.model.get_submodule("head").training

    def test__eval__should_put_every_submodule_in_eval_mode(self, frozen_head: HFHead) -> None:
        """Evaluation mode reaches trainable and frozen submodules alike."""
        frozen_head.train()

        frozen_head.eval()

        assert not any(submodule.training for submodule in frozen_head.model.modules())
