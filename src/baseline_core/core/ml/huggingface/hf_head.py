"""Heads wrapping a Hugging Face model that owns its architecture and loss."""

from typing import Any, Self, override

import torch
import transformers
from pydantic import Field, field_validator
from transformers import PreTrainedModel

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.dynamic_import import DynamicImport
from baseline_core.core.ml.model.abstract_head import AbstractHead, HeadConfig, LossResult


class HFHeadConfig(HeadConfig):
    """Which Hugging Face model and processor to load, and how.

    Attributes:
        model_id: Hugging Face id or local directory.
        auto_class: ``transformers`` auto class loading the model (``"AutoModelForCausalLM"``).
        processor_class: ``transformers`` class loading the processor.
        model_kwargs: Keyword arguments of the model's ``from_pretrained``.
        processor_kwargs: Keyword arguments of the processor's ``from_pretrained``.
        freeze: Parameter-name prefixes to freeze, e.g. ``["model.backbone"]``.
    """

    model_id: str
    auto_class: str
    processor_class: str = "AutoProcessor"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    processor_kwargs: dict[str, Any] = Field(default_factory=dict)
    freeze: list[str] = Field(default_factory=list)

    @field_validator("auto_class", "processor_class")
    @classmethod
    def validate_class(cls, value: str) -> str:
        """Fail fast when ``transformers`` has no such class."""
        try:
            DynamicImport.resolve_class(transformers, value)
        except ImportError as error:
            error_msg = f"transformers has no class '{value}'."
            raise ValueError(error_msg) from error

        return value


class HFHead(AbstractHead[Batch[Any], dict[str, torch.Tensor]]):
    """A Hugging Face ml model trained with the loss it computes from ``labels``.

    Attributes:
        config: Model and processor settings.
        model: The Hugging Face ml model.
        processor: Prepares the model inputs and post-processes its outputs.
    """

    config: HFHeadConfig

    def __init__(self, config: HFHeadConfig) -> None:
        """Load the model and its processor from ``config.model_id``, then freeze parameters.

        Args:
            config: Model and processor settings.
        """
        super().__init__(config)
        self.model = self._load_model(config)
        self.processor = self._load_processor(config)
        self._freeze(config.freeze)

    @override
    def train(self, mode: bool = True) -> Self:
        """Set ``mode`` everywhere but in frozen submodules, which stay in evaluation mode.

        Args:
            mode: Training mode when ``True``, evaluation mode when ``False``.

        Returns:
            The module itself.
        """
        super().train(mode)

        if mode:
            frozen_prefixes = tuple(self.config.freeze)

            for name, submodule in self.model.named_modules():
                if frozen_prefixes and name.startswith(frozen_prefixes):
                    submodule.eval()

        return self

    @override
    def forward(self, features: dict[str, torch.Tensor], batch: Batch[Any]) -> Any:
        """Run the model on ``features``, with the batch targets as ``labels`` when there are some.

        Args:
            features: Model inputs, as the processor prepared them.
            batch: Batch the inputs come from.

        Returns:
            The model outputs, with a ``loss`` when the batch has targets.
        """
        if batch.targets is None:
            return self.model(**features)

        return self.model(**features, labels=batch.targets)

    @override
    def loss(self, predictions: Any, batch: Batch[Any]) -> LossResult:
        """The loss the model computed, with its components.

        Args:
            predictions: Model outputs of ``batch``.
            batch: Batch the model ran on.

        Returns:
            The loss, and the ``loss_dict`` entries of the outputs as extra fields.

        Raises:
            ValueError: If the batch has no targets.
        """
        if batch.targets is None:
            error_msg = "Cannot compute a loss on a batch without targets."
            raise ValueError(error_msg)

        return LossResult(loss=predictions.loss, **(getattr(predictions, "loss_dict", None) or {}))

    @override
    def metric_args(self, predictions: Any, batch: Batch[Any]) -> tuple[Any, Any]:
        """The postprocessed outputs and targets.

        Args:
            predictions: Model outputs of ``batch``.
            batch: Batch the model ran on.

        Returns:
            What ``postprocess`` and ``postprocess_targets`` return.
        """
        return self.postprocess(predictions, batch), self.postprocess_targets(batch.targets)

    def postprocess(self, outputs: Any, batch: Batch[Any]) -> Any:  # noqa: ANN401, PLR6301
        """Turn the model outputs into task predictions; outputs unchanged by default.

        Args:
            outputs: Model outputs of ``batch``.
            batch: Batch the model ran on.

        Returns:
            The predictions.
        """
        del batch
        return outputs

    def postprocess_targets(self, targets: Any) -> Any:  # noqa: ANN401, PLR6301
        """Turn the batch targets into the ground truth the metrics compare predictions with.

        Args:
            targets: Targets of a batch, as the collator built them.

        Returns:
            The ground truth; the targets unchanged by default.
        """
        return targets

    @staticmethod
    def _load_model(config: HFHeadConfig) -> PreTrainedModel:
        """Load the pretrained model with ``config.auto_class``."""
        auto_class = DynamicImport.resolve_class(transformers, config.auto_class)
        model: PreTrainedModel = auto_class.from_pretrained(
            config.model_id, use_safetensors=True, trust_remote_code=False, **config.model_kwargs
        )
        return model

    @staticmethod
    def _load_processor(config: HFHeadConfig) -> Any:  # noqa: ANN401
        """Load the pretrained processor with ``config.processor_class``."""
        processor_class = DynamicImport.resolve_class(transformers, config.processor_class)
        return processor_class.from_pretrained(
            config.model_id, trust_remote_code=False, **config.processor_kwargs
        )

    def _freeze(self, prefixes: list[str]) -> None:
        """Stop gradients of the parameters whose name starts with one of ``prefixes``.

        Raises:
            ValueError: If a prefix matches no parameter.
        """
        unmatched = set(prefixes)

        for name, parameter in self.model.named_parameters():
            matched = {prefix for prefix in prefixes if name.startswith(prefix)}

            if matched:
                parameter.requires_grad = False
                unmatched -= matched

        if unmatched:
            error_msg = f"Freeze prefixes match no parameter: {sorted(unmatched)}"
            raise ValueError(error_msg)
