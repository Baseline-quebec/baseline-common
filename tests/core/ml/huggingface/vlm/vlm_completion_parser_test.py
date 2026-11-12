"""Tests for ``VlmCompletionParser``."""

import pytest

from baseline_core.core.ml.huggingface.vlm.vlm_completion_parser import VlmCompletionParser
from baseline_core.core.ml.modality.image.annotation import BoundingBox, BoxAnnotation


class TestVlmCompletionParser:
    """Tests for reading completions in both formats."""

    @pytest.fixture
    def parser(self) -> VlmCompletionParser:
        """The parser."""
        return VlmCompletionParser()

    @pytest.fixture
    def labels(self) -> list[str]:
        """Labels the prompt asked for."""
        return ["box", "bag"]

    def test__parse__should_scale_the_grid_to_the_image(
        self, parser: VlmCompletionParser, labels: list[str]
    ) -> None:
        """Grid coordinates become pixels of a 200x100 image."""
        completion = "box 100 200 500 800\n"

        boxes = parser.parse(completion, labels, width=200, height=100)

        assert boxes == [BoxAnnotation(label="box", box=BoundingBox(x1=20, y1=20, x2=100, y2=80))]

    def test__parse__should_read_the_json_format_too(
        self, parser: VlmCompletionParser, labels: list[str]
    ) -> None:
        """The JSON format Qwen models were trained on is accepted."""
        completion = '[{"bbox_2d": [0, 0, 500, 500], "label": "bag"}]'

        boxes = parser.parse(completion, labels, width=100, height=100)

        assert boxes == [BoxAnnotation(label="bag", box=BoundingBox(x1=0, y1=0, x2=50, y2=50))]

    def test__parse__should_drop_repeated_lines_unknown_labels_and_empty_boxes(
        self, parser: VlmCompletionParser, labels: list[str]
    ) -> None:
        """A model looping on one line, an unasked label and a zero-area box give nothing extra."""
        completion = "bag 10 10 50 50\nbag 10 10 50 50\ncrate 0 0 100 100\nbag 20 20 20 60\n"

        boxes = parser.parse(completion, labels, width=100, height=100)

        assert [box.label for box in boxes] == ["bag"]

    def test__parse__should_clamp_coordinates_to_the_grid(
        self, parser: VlmCompletionParser, labels: list[str]
    ) -> None:
        """Values past 1000 end at the image border."""
        boxes = parser.parse("bag 900 900 1200 1500", labels, width=100, height=100)

        assert boxes[0].box.xyxy == [90, 90, 100, 100]

    @pytest.mark.parametrize("completion", ["none", "", "I see a box."])
    def test__parse__when_no_box__should_return_nothing(
        self, parser: VlmCompletionParser, labels: list[str], completion: str
    ) -> None:
        """A completion without a parsable line yields no box, not an error."""
        assert parser.parse(completion, labels, width=100, height=100) == []
