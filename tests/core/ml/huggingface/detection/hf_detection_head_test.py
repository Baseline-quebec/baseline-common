"""Tests for the Hugging Face detection head and its config."""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture
from transformers.utils.generic import ModelOutput

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.data.phase import Phase
from baseline_core.core.ml.huggingface.detection.hf_detection_head import (
    HFDetectionConfig,
    HFDetectionHead,
)
from baseline_core.core.ml.modality.image.annotation import ImageAnnotation
from baseline_core.core.ml.training.backbone_heads_module import BackboneHeadsModule


class TestHFDetectionConfig:
    """Tests for ``HFDetectionConfig``."""

    def test__init__should_default_to_the_object_detection_auto_class(self) -> None:
        """Only the model id is needed for a detector."""
        config = HFDetectionConfig(model_id="dummy")

        assert config.auto_class == "AutoModelForObjectDetection"
        assert config.target == "detection"


class TestHFDetectionHead:
    """Tests for ``HFDetectionHead`` around the tiny RF-DETR."""

    @pytest.fixture
    def images(self, collated_batch: Batch[Any]) -> list[ImageAnnotation]:
        """The annotated images of the batch."""
        assert collated_batch.metadata is not None
        annotations: list[ImageAnnotation] = collated_batch.metadata["annotations"]
        return annotations

    @pytest.fixture
    def outputs(self, detection_head: HFDetectionHead, collated_batch: Batch[Any]) -> ModelOutput:
        """Model outputs of the batch, with the losses its targets give."""
        outputs: ModelOutput = detection_head(collated_batch.x, collated_batch)
        return outputs

    @pytest.fixture
    def strict_head(self, detection_config: HFDetectionConfig) -> HFDetectionHead:
        """The same detector keeping only predictions of full confidence, i.e. none."""
        return HFDetectionHead(detection_config.model_copy(update={"threshold": 1.0}))

    @pytest.fixture
    def empty_target(self, empty_image: ImageAnnotation) -> dict[str, torch.Tensor]:
        """The processor target of an image without any box."""
        return {
            "class_labels": torch.zeros(0, dtype=torch.long),
            "boxes": torch.zeros(0, 4),
            "orig_size": torch.tensor([empty_image.height, empty_image.width]),
        }

    @pytest.fixture
    def log_dict(self, detection_module: BackboneHeadsModule, mocker: MockerFixture) -> MagicMock:
        """``module.log_dict`` replaced by a mock: there is no trainer to receive the values."""
        return mocker.patch.object(detection_module, "log_dict")

    def test__loss__should_return_the_loss_and_its_components(
        self, detection_head: HFDetectionHead, collated_batch: Batch[Any], outputs: ModelOutput
    ) -> None:
        """RF-DETR reports its classification and box losses."""
        result = detection_head.loss(outputs, collated_batch)

        assert result.loss.ndim == 0
        assert {"loss_ce", "loss_bbox", "loss_giou"} <= set(result.model_extra or {})

    def test__id2label__should_follow_the_model_config(
        self, detection_head: HFDetectionHead, labels: list[str]
    ) -> None:
        """Label names come from the model, in id order."""
        assert detection_head.id2label == dict(enumerate(labels))

    def test__postprocess__should_return_scored_boxes_per_image(
        self,
        detection_head: HFDetectionHead,
        collated_batch: Batch[Any],
        outputs: ModelOutput,
        images: list[ImageAnnotation],
    ) -> None:
        """One result per image, with as many boxes, labels and scores."""
        predictions = detection_head.postprocess(outputs, collated_batch)

        assert len(predictions) == len(images)
        assert all(len(predictions[0][key]) for key in ("boxes", "labels", "scores"))
        assert all(
            prediction["boxes"].shape == (len(prediction["scores"]), 4)
            for prediction in predictions
        )

    def test__postprocess__should_drop_boxes_below_the_threshold(
        self, strict_head: HFDetectionHead, collated_batch: Batch[Any]
    ) -> None:
        """No prediction reaches full confidence."""
        outputs = strict_head(collated_batch.x, collated_batch)

        predictions = strict_head.postprocess(outputs, collated_batch)

        assert not any(len(prediction["boxes"]) for prediction in predictions)

    def test__metric_args__should_keep_the_boxes_below_the_threshold(
        self, strict_head: HFDetectionHead, collated_batch: Batch[Any]
    ) -> None:
        """Mean average precision sees every box, whatever the output threshold."""
        outputs = strict_head(collated_batch.x, collated_batch)

        predictions, _ = strict_head.metric_args(outputs, collated_batch)

        assert all(len(prediction["boxes"]) for prediction in predictions)

    def test__postprocess__when_no_targets__should_raise_value_error(
        self, detection_head: HFDetectionHead, collated_batch: Batch[Any], outputs: ModelOutput
    ) -> None:
        """Image sizes come from the targets."""
        bare_batch = collated_batch.model_copy(update={"targets": None})

        with pytest.raises(ValueError, match="orig_size"):
            detection_head.postprocess(outputs, bare_batch)

    def test__postprocess_targets__should_give_pixel_xyxy_boxes_and_label_ids(
        self,
        detection_head: HFDetectionHead,
        collated_batch: Batch[Any],
        images: list[ImageAnnotation],
        labels: list[str],
    ) -> None:
        """The ground truth matches the annotated boxes, in pixels."""
        assert collated_batch.targets is not None

        truth = detection_head.postprocess_targets(collated_batch.targets)

        for image, image_truth in zip(images, truth, strict=True):
            expected_boxes = torch.tensor([box.box.xyxy for box in image.boxes])
            expected_labels = [labels.index(box.label) for box in image.boxes]
            assert torch.allclose(image_truth["boxes"], expected_boxes, atol=1e-3)
            assert image_truth["labels"].tolist() == expected_labels

    def test__postprocess_targets__when_no_box__should_return_empty_tensors(
        self, detection_head: HFDetectionHead, empty_target: dict[str, torch.Tensor]
    ) -> None:
        """An empty image still has a ``(0, 4)`` box tensor, as ``torchmetrics`` requires."""
        (truth,) = detection_head.postprocess_targets([empty_target])

        assert tuple(truth["boxes"].shape) == (0, 4)
        assert tuple(truth["labels"].shape) == (0,)

    def test__on_validation_epoch_end__should_log_scalar_metrics_then_reset(
        self,
        detection_module: BackboneHeadsModule,
        detection_config: HFDetectionConfig,
        collated_batch: Batch[Any],
        log_dict: MagicMock,
    ) -> None:
        """Validation logs mAP values, keeps per-class tensors out, starts the next epoch clean."""
        output = detection_module.step(collated_batch)
        detection_module.update_metrics(Phase.VAL, output, collated_batch)
        collection = detection_module.metrics_for(Phase.VAL)[detection_config.target]

        detection_module.on_validation_epoch_end()

        logged = log_dict.call_args.args[0]
        assert {"val_detection_map", "val_detection_map_50"} <= set(logged)
        assert not any(name.endswith(("per_class", "classes")) for name in logged)
        assert all(value.numel() == 1 for value in logged.values())
        assert collection["MeanAveragePrecision"].detection_box == []
