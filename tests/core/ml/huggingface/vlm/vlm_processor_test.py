"""Tests for ``VlmProcessor``."""

import pytest
import torch
from PIL import Image

from baseline_core.core.ml.huggingface.vlm.vlm_processor import VlmProcessor
from tests.core.ml.huggingface.vlm.dummy_classes import FakeHFProcessor


class TestVlmProcessor:
    """Tests for preparing inputs and decoding completions."""

    @pytest.fixture
    def processor(self, fake_hf_processor: FakeHFProcessor) -> VlmProcessor:
        """Processor sending at most 10 000 pixels."""
        return VlmProcessor(fake_hf_processor, max_pixels=10_000)

    def test__init__when_max_pixels_not_positive__should_raise_value_error(
        self, fake_hf_processor: FakeHFProcessor
    ) -> None:
        """A zero pixel budget is a configuration error."""
        with pytest.raises(ValueError, match="max_pixels"):
            VlmProcessor(fake_hf_processor, max_pixels=0)

    def test__prepare__should_send_the_prompt_through_the_chat_template(
        self, processor: VlmProcessor, fake_hf_processor: FakeHFProcessor
    ) -> None:
        """The prompt is one user turn with the image, generation prompt added, no thinking."""
        inputs = processor.prepare(Image.new("RGB", (50, 50)), "Find the objects.")

        assert fake_hf_processor.template_kwargs == {
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        assert (
            fake_hf_processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            == "<user>Find the objects.<assistant>"
        )

    def test__prepare__when_image_too_large__should_downscale_it_keeping_the_aspect_ratio(
        self, processor: VlmProcessor, fake_hf_processor: FakeHFProcessor
    ) -> None:
        """A 400x100 image becomes 200x50 under the 10 000 pixel budget."""
        processor.prepare(Image.new("RGB", (400, 100)), "Find the objects.")

        assert fake_hf_processor.sent_images[0].size == (200, 50)

    def test__prepare__when_image_small_enough__should_send_it_as_is(
        self, processor: VlmProcessor, fake_hf_processor: FakeHFProcessor
    ) -> None:
        """Images under the budget are not resized."""
        image = Image.new("RGB", (80, 60))

        processor.prepare(image, "Find the objects.")

        assert fake_hf_processor.sent_images[0] is image

    def test__decode__should_read_the_tokens_after_the_prompt(
        self, processor: VlmProcessor
    ) -> None:
        """Only the generated part of the first sequence is decoded."""
        sequences = torch.tensor([[ord(character) for character in "promptbag 1 2 3 4"]])

        assert processor.decode(sequences, prompt_length=6) == "bag 1 2 3 4"
