"""Tests for ``VlmPromptBuilder``."""

import pytest

from baseline_core.core.ml.data.taxonomy import Taxonomy
from baseline_core.core.ml.huggingface.vlm.vlm_prompt_builder import VlmPromptBuilder


class TestVlmPromptBuilder:
    """Tests for the prompt text."""

    def test__build__should_list_every_label_with_its_description(
        self, taxonomy: Taxonomy
    ) -> None:
        """The context opens the prompt; each label gets one line with its description."""
        prompt = VlmPromptBuilder(taxonomy).build(["box", "bag"], "Find the objects.")

        assert prompt.text.startswith("Find the objects.")
        assert "- box: A shut box." in prompt.text
        assert "- bag: One bag." in prompt.text
        assert "label x1 y1 x2 y2" in prompt.text
        assert prompt.labels == ["box", "bag"]

    def test__build__when_label_unknown__should_raise_value_error(
        self, taxonomy: Taxonomy
    ) -> None:
        """A label missing from the taxonomy has no description to give the model."""
        with pytest.raises(ValueError, match=r"\['crate'\] are not in the taxonomy"):
            VlmPromptBuilder(taxonomy).build(["crate"], "Find the objects.")

    def test__build__when_no_label__should_raise_value_error(self, taxonomy: Taxonomy) -> None:
        """A prompt asks for at least one label."""
        with pytest.raises(ValueError, match="labels"):
            VlmPromptBuilder(taxonomy).build([], "Find the objects.")
