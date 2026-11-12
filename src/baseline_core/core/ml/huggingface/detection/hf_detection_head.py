"""Head wrapping a Hugging Face object detection model."""

from typing import Any, ClassVar, override

import torch
from torchvision.ops import box_convert

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.huggingface.hf_head import HFHead, HFHeadConfig


class HFDetectionConfig(HFHeadConfig):
    """Which Hugging Face detection model to train.

    Attributes:
        target: Key of the detector's predictions, losses and metrics.
        auto_class: ``transformers`` auto class loading the model.
        threshold: Minimum confidence of a returned box; the metrics see every box.
    """

    target: str = "detection"
    auto_class: str = "AutoModelForObjectDetection"
    threshold: float = 0.3


class HFDetectionHead(HFHead):
    """A Hugging Face detector trained with its own loss, its boxes fed to the metrics in pixels.

    Attributes:
        config: Model and processor settings.
    """

    METRIC_THRESHOLD: ClassVar[float] = 0.0

    config: HFDetectionConfig

    @property
    def id2label(self) -> dict[int, str]:
        """Label of every model id, as the model config holds them."""
        return {
            int(label_id): str(label)
            for label_id, label in (self.model.config.id2label or {}).items()
        }

    @override
    def metric_args(self, predictions: Any, batch: Batch[Any]) -> tuple[Any, Any]:
        """Every scored box, whatever its confidence, and the ground truth, in pixels.

        Args:
            predictions: Model outputs of ``batch``.
            batch: Batch the model ran on.

        Returns:
            The boxes from ``METRIC_THRESHOLD`` and what ``postprocess_targets`` returns.

        Raises:
            ValueError: If the batch has no targets.
        """
        if batch.targets is None:
            error_msg = "Metrics compare predictions with the targets; the batch has none."
            raise ValueError(error_msg)

        return (
            self.scored_boxes(predictions, batch, self.METRIC_THRESHOLD),
            self.postprocess_targets(batch.targets),
        )

    @override
    def postprocess(self, outputs: Any, batch: Batch[Any]) -> list[dict[str, torch.Tensor]]:
        """The boxes returned to callers: ``scored_boxes`` from ``config.threshold``.

        Args:
            outputs: Model outputs of ``batch``.
            batch: Batch the model ran on; its targets give the image sizes.

        Returns:
            Per image, ``boxes`` in pixel ``xyxy``, label ids as ``labels`` and ``scores``.
        """
        return self.scored_boxes(outputs, batch, self.config.threshold)

    def scored_boxes(
        self,
        outputs: Any,  # noqa: ANN401
        batch: Batch[Any],
        threshold: float,
    ) -> list[dict[str, torch.Tensor]]:
        """Scored boxes of every image of ``batch`` in pixel ``xyxy``, from ``threshold``.

        Args:
            outputs: Model outputs of ``batch``.
            batch: Batch the model ran on; its targets give the image sizes.
            threshold: Minimum confidence of a kept box.

        Returns:
            Per image, ``boxes`` in pixel ``xyxy``, label ids as ``labels`` and ``scores``.

        Raises:
            ValueError: If the batch has no targets.
        """
        if batch.targets is None:
            error_msg = "Image sizes come from the targets' 'orig_size'; the batch has none."
            raise ValueError(error_msg)

        target_sizes = torch.stack([target["orig_size"] for target in batch.targets])
        results: list[dict[str, torch.Tensor]] = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )
        return results

    @override
    def postprocess_targets(
        self, targets: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """The ground truth of every image as ``torchmetrics`` expects it.

        Args:
            targets: Per image, ``class_labels``, normalized ``cxcywh`` boxes, ``orig_size``.

        Returns:
            Per image, ``boxes`` in pixel ``xyxy`` and their ``labels``.
        """
        return [self.to_pixel_target(target) for target in targets]

    @staticmethod
    def to_pixel_target(target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The ground truth of one image as ``torchmetrics`` expects it: pixel ``xyxy`` boxes.

        Args:
            target: ``class_labels``, normalized ``cxcywh`` boxes and ``orig_size`` of one image.

        Returns:
            ``boxes`` in pixel ``xyxy`` and their ``labels``.
        """
        height, width = target["orig_size"].tolist()
        scale = target["boxes"].new_tensor([width, height, width, height])
        boxes = box_convert(target["boxes"], in_fmt="cxcywh", out_fmt="xyxy") * scale
        return {"boxes": boxes, "labels": target["class_labels"]}
