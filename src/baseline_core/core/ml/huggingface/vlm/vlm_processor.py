"""Preparing an image and a prompt for a chat vision-language model, decoding its completion."""

import math
from typing import Any, ClassVar

import torch
from PIL import Image


class VlmProcessor:
    """Wraps the Hugging Face processor of a chat vision-language model.

    Attributes:
        INPUT_IDS_KEY: Name of the prompt token ids among the prepared inputs.
        processor: The model's processor: tokenizer, image processor and chat template.
        max_pixels: Largest image sent; bigger ones are downscaled, keeping the aspect ratio.
    """

    INPUT_IDS_KEY: ClassVar[str] = "input_ids"

    def __init__(self, processor: Any, max_pixels: int) -> None:  # noqa: ANN401
        """Keep the processor and the pixel budget.

        Args:
            processor: The model's Hugging Face processor.
            max_pixels: Largest image sent to the model.

        Raises:
            ValueError: If ``max_pixels`` is not positive.
        """
        if max_pixels <= 0:
            error_msg = f"max_pixels must be positive, got {max_pixels}."
            raise ValueError(error_msg)

        self.processor = processor
        self.max_pixels = max_pixels

    def prepare(self, image: Image.Image, prompt: str) -> dict[str, torch.Tensor]:
        """Build the model inputs of one user turn holding ``image`` and ``prompt``.

        Args:
            image: RGB image.
            prompt: Text of the turn.

        Returns:
            The input tensors by name, ``input_ids`` among them.
        """
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        # ``enable_thinking`` turns off the reasoning preamble of Qwen3 templates; templates
        # without that variable ignore it.
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(
            images=self.resized(image, self.max_pixels), text=text, return_tensors="pt"
        )
        return dict(inputs)

    def decode(self, sequences: torch.Tensor, prompt_length: int) -> str:
        """Decode the completion of the first sequence, after its ``prompt_length`` tokens.

        Args:
            sequences: Generated token ids, prompt included, one row per sequence.
            prompt_length: Number of prompt tokens at the start of every sequence.

        Returns:
            The completion text, without special tokens.
        """
        completion: str = self.processor.tokenizer.decode(
            sequences[0, prompt_length:], skip_special_tokens=True
        )
        return completion

    @staticmethod
    def resized(image: Image.Image, max_pixels: int) -> Image.Image:
        """Downscale ``image`` to at most ``max_pixels``, keeping its aspect ratio.

        Args:
            image: Image to send.
            max_pixels: Largest number of pixels.

        Returns:
            ``image`` itself when small enough, a downscaled copy otherwise.
        """
        scale = math.sqrt(max_pixels / (image.width * image.height))

        if scale >= 1:
            return image

        return image.resize((round(image.width * scale), round(image.height * scale)))
