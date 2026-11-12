"""Tests for ``VlmDetector`` on a fake model echoing a fixed completion."""

import pytest
from PIL import Image

from baseline_core.core.ml.huggingface.vlm.vlm_completion_parser import VlmCompletionParser
from baseline_core.core.ml.huggingface.vlm.vlm_detector import VlmDetector
from baseline_core.core.ml.huggingface.vlm.vlm_processor import VlmProcessor
from baseline_core.core.ml.huggingface.vlm.vlm_prompt_builder import Prompt
from baseline_core.core.ml.modality.image.annotation import BoundingBox, BoxAnnotation
from baseline_core.core.ml.model_runner import TorchModelRunner
from tests.core.ml.huggingface.vlm.dummy_classes import EchoModel, FakeHFProcessor


class TestVlmDetector:
    """Tests for the prompt-to-boxes round trip."""

    @pytest.fixture
    def completion(self) -> str:
        """A completion with one box of each label."""
        return "box 0 0 500 500\nbag 500 500 1000 1000\n"

    @pytest.fixture
    def detector(self, completion: str, fake_hf_processor: FakeHFProcessor) -> VlmDetector:
        """Detector whose model generates ``completion`` after any prompt."""
        model = EchoModel(completion)
        runner = TorchModelRunner(
            model, output_names=[VlmDetector.SEQUENCES_KEY], call=model.generate
        )
        return VlmDetector(
            runner, VlmProcessor(fake_hf_processor, max_pixels=1_000_000), VlmCompletionParser()
        )

    @pytest.fixture
    def prompt(self) -> Prompt:
        """Prompt asking for both labels."""
        return Prompt(text="Find the objects.", labels=["box", "bag"])

    def test__call__should_return_the_boxes_in_image_pixels(
        self, detector: VlmDetector, prompt: Prompt
    ) -> None:
        """The grid of the completion is scaled to the image."""
        image = Image.new("RGB", (200, 100))

        detection = detector(image, prompt)

        assert detection.boxes == [
            BoxAnnotation(label="box", box=BoundingBox(x1=0, y1=0, x2=100, y2=50)),
            BoxAnnotation(label="bag", box=BoundingBox(x1=100, y1=50, x2=200, y2=100)),
        ]

    def test__call__should_keep_only_the_labels_of_the_prompt(self, detector: VlmDetector) -> None:
        """The same model and completion give different boxes under a narrower prompt."""
        prompt = Prompt(text="Find the bags.", labels=["bag"])

        detection = detector(Image.new("RGB", (10, 10)), prompt)

        assert [box.label for box in detection.boxes] == ["bag"]

    def test__call__should_return_the_completion_and_its_time(
        self, detector: VlmDetector, prompt: Prompt, completion: str
    ) -> None:
        """The completion comes back as decoded, with a positive duration."""
        detection = detector(Image.new("RGB", (10, 10)), prompt)

        assert detection.completion.text == completion
        assert detection.completion.seconds > 0
