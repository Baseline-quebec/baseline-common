"""Stand-ins for the Hugging Face processor and model of a chat vision-language model."""

from typing import Any

import torch
from PIL import Image


class FakeTokenizer:
    """Tokens are character codes, so decoding is reading them back."""

    @staticmethod
    def decode(ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        """The characters of ``ids``."""
        del skip_special_tokens
        return "".join(chr(int(code)) for code in ids.tolist())


class FakeHFProcessor:
    """Records the chat template call and tokenizes the text character by character.

    Attributes:
        tokenizer: Decodes token ids back to text.
        sent_images: Images given to the processor, in call order.
        template_kwargs: Keyword arguments of the last ``apply_chat_template`` call.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.tokenizer = FakeTokenizer()
        self.sent_images: list[Image.Image] = []
        self.template_kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: object) -> str:
        """The text of the user turn, prefixed as a chat template would."""
        self.template_kwargs = kwargs
        text = next(part["text"] for part in messages[0]["content"] if part["type"] == "text")
        return f"<user>{text}<assistant>"

    def __call__(
        self, images: Image.Image, text: str, return_tensors: str
    ) -> dict[str, torch.Tensor]:
        """Character codes of ``text`` as ``input_ids``, plus a dummy pixel tensor."""
        del return_tensors
        self.sent_images.append(images)
        input_ids = torch.tensor([[ord(character) for character in text]])
        return {"input_ids": input_ids, "pixel_values": torch.zeros(1, 3, 4, 4)}


class EchoModel(torch.nn.Module):
    """Generates a fixed completion after the prompt, as character codes.

    Attributes:
        completion: Text generated after every prompt.
    """

    def __init__(self, completion: str) -> None:
        """Keep the completion; one parameter gives the module a device."""
        super().__init__()
        self.completion = completion
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """The prompt ids followed by the completion's character codes."""
        del kwargs
        generated = torch.tensor([[ord(character) for character in self.completion]])
        return torch.cat([input_ids, generated], dim=1)
