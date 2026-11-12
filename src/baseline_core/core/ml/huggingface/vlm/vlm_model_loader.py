"""Loading a vision-language model and its processor from the Hugging Face hub."""

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


class VlmModelConfig(BaseModel):
    """Which vision-language model to load and how.

    Attributes:
        model_id: Hugging Face id or local directory.
        load_kwargs: Keyword arguments of ``from_pretrained``, e.g. ``dtype`` and ``device_map``.
        quantization: ``BitsAndBytesConfig`` keyword arguments, or ``None`` for stored weights.
    """

    model_id: str
    load_kwargs: dict[str, Any] = Field(default_factory=dict)
    quantization: dict[str, Any] | None = None


class VlmModelLoader:
    """Loads the model of a config, safetensors only and without remote code, and its processor.

    Attributes:
        config: Which model to load and how.
    """

    def __init__(self, config: VlmModelConfig) -> None:
        """Keep the config.

        Args:
            config: Which model to load and how.
        """
        self.config = config

    def load(self) -> tuple[Any, Any]:
        """Load the model in evaluation mode and its processor.

        Returns:
            The model and its processor.
        """
        load_kwargs = self.load_kwargs()
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            self.config.model_id, trust_remote_code=False
        )
        model = AutoModelForImageTextToText.from_pretrained(  # type: ignore[no-untyped-call]
            self.config.model_id, trust_remote_code=False, use_safetensors=True, **load_kwargs
        ).eval()
        logger.info("Loaded {} with {}", self.config.model_id, load_kwargs)
        return model, processor

    def load_kwargs(self) -> dict[str, Any]:
        """The ``from_pretrained`` keyword arguments, quantization config built.

        Returns:
            ``config.load_kwargs``, plus ``quantization_config`` when the config quantizes.
        """
        load_kwargs = dict(self.config.load_kwargs)

        if self.config.quantization is not None:
            options = self.config.quantization
            quantization = BitsAndBytesConfig(**options)  # type: ignore[no-untyped-call]
            load_kwargs["quantization_config"] = quantization

        return load_kwargs
