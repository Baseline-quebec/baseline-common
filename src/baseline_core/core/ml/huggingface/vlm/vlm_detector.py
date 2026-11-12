"""A vision-language model asked for the boxes of a prompt's labels, one image at a time."""

import time
from typing import ClassVar

from PIL import Image
from pydantic import BaseModel, Field

from baseline_core.core.ml.huggingface.vlm.vlm_completion_parser import VlmCompletionParser
from baseline_core.core.ml.huggingface.vlm.vlm_processor import VlmProcessor
from baseline_core.core.ml.huggingface.vlm.vlm_prompt_builder import Prompt
from baseline_core.core.ml.modality.image.annotation import BoxAnnotation
from baseline_core.core.ml.model_runner import AbstractModelRunner


class Completion(BaseModel):
    """What the model generated for one prompt.

    Attributes:
        text: The completion text.
        seconds: Generation time.
    """

    text: str
    seconds: float = Field(ge=0)


class VlmDetection(BaseModel):
    """The boxes read from one completion, with the completion itself.

    Attributes:
        boxes: The boxes, in the image's pixel coordinates, without scores.
        completion: What the model generated, kept for inspection.
    """

    boxes: list[BoxAnnotation]
    completion: Completion


class VlmDetector:
    """Finds the boxes a prompt asks for with a chat vision-language model, on any image.

    Attributes:
        SEQUENCES_KEY: Name the runner gives the generated token ids.
        runner: Generates with the model; built with ``output_names=[SEQUENCES_KEY]``.
        processor: Prepares the image and the prompt, decodes the completion.
        parser: Turns the completion into boxes; the only piece bound to the model family.
    """

    SEQUENCES_KEY: ClassVar[str] = "sequences"

    def __init__(
        self, runner: AbstractModelRunner, processor: VlmProcessor, parser: VlmCompletionParser
    ) -> None:
        """Keep the collaborators.

        Args:
            runner: Generates with the model.
            processor: Prepares the image and the prompt, decodes the completion.
            parser: Turns the completion into boxes.
        """
        self.runner = runner
        self.processor = processor
        self.parser = parser

    def __call__(self, image: Image.Image, prompt: Prompt) -> VlmDetection:
        """Ask ``prompt`` about ``image``.

        Args:
            image: RGB image.
            prompt: What to ask, and which labels to keep from the completion.

        Returns:
            The boxes, none when the completion does not parse, and the completion.
        """
        inputs = self.processor.prepare(image, prompt.text)
        prompt_length = inputs[self.processor.INPUT_IDS_KEY].shape[1]
        start = time.perf_counter()
        outputs = self.runner(inputs)
        seconds = time.perf_counter() - start
        text = self.processor.decode(outputs[self.SEQUENCES_KEY], prompt_length)
        return VlmDetection(
            boxes=self.parser.parse(text, prompt.labels, image.width, image.height),
            completion=Completion(text=text, seconds=seconds),
        )
