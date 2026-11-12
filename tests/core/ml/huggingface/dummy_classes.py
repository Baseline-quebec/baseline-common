"""Minimal Hugging Face models and processor for the adapter, detection and service tests."""

from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import (
    PreTrainedConfig,
    PreTrainedModel,
    RfDetrConfig,
    RfDetrForObjectDetection,
    RfDetrImageProcessor,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.rf_detr.configuration_rf_detr import RfDetrDinov2Config


class DummyHFConfig(PreTrainedConfig):  # type: ignore[no-untyped-call]  # untyped hub base
    """Configuration of ``DummyHFModel``."""

    model_type = "dummy_hf"
    input_size: int = 4
    hidden_size: int = 8


class DummyHFModel(PreTrainedModel):  # type: ignore[no-untyped-call]  # untyped hub base
    """Hugging Face model regressing one value per sample, with its own loss."""

    config_class = DummyHFConfig

    def __init__(self, config: DummyHFConfig) -> None:
        """Create an encoder and a head."""
        super().__init__(config)
        self.encoder = nn.Linear(config.input_size, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, 1)
        self.post_init()  # type: ignore[no-untyped-call]  # transformers leaves it unannotated

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None
    ) -> SequenceClassifierOutput:
        """Predict, and compute the squared error when ``labels`` are given."""
        logits = self.head(self.encoder(pixel_values))
        loss = ((logits - labels) ** 2).mean() if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


class StackingProcessor:
    """Stand-in for a Hugging Face processor: stacks images, passes annotations through."""

    def __init__(self) -> None:
        """Record the keyword arguments of the last call."""
        self.last_kwargs: dict[str, Any] = {}

    def __call__(
        self,
        images: list[torch.Tensor],
        annotations: list[dict[str, Any]],
        **kwargs: Any,  # noqa: ANN401
    ) -> Mapping[str, Any]:
        """Return pixel values and the per-image supervision."""
        self.last_kwargs = kwargs
        return {
            "pixel_values": torch.stack(images),
            "pixel_mask": torch.ones(len(images), dtype=torch.bool),
            "labels": annotations,
        }


class TinyRfDetr:
    """The RF-DETR the detection tests run: random weights, small enough for a CPU."""

    IMAGE_SIZE: ClassVar[int] = 64

    @staticmethod
    def model(labels: list[str]) -> RfDetrForObjectDetection:
        """A randomly initialized RF-DETR over ``labels``, in eval mode.

        Args:
            labels: Label names, in category order.

        Returns:
            The model, seeded so every build has the same weights.
        """
        # out_features reaches __post_init__ through **kwargs, which mypy does not see.
        backbone = RfDetrDinov2Config(  # type: ignore[call-arg]
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            out_features=["stage1", "stage2", "stage3", "stage4"],
            image_size=TinyRfDetr.IMAGE_SIZE,
            patch_size=16,
            num_windows=1,
        )
        config = RfDetrConfig(
            backbone_config=backbone,
            d_model=32,
            decoder_layers=1,
            decoder_self_attention_heads=2,
            decoder_cross_attention_heads=2,
            decoder_ffn_dim=32,
            num_queries=10,
            group_detr=2,
            intermediate_size=32,
            id2label=dict(enumerate(labels)),
            label2id={label: label_id for label_id, label in enumerate(labels)},
        )
        torch.manual_seed(0)
        model = RfDetrForObjectDetection(config)
        model.train(mode=False)
        return model

    @staticmethod
    def processor() -> RfDetrImageProcessor:
        """The RF-DETR processor, resizing to the model's input.

        Returns:
            The processor.
        """
        size = TinyRfDetr.IMAGE_SIZE
        return RfDetrImageProcessor(size={"height": size, "width": size})
