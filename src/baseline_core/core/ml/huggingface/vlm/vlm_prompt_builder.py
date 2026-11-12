"""The prompt asking a vision-language model for one box per object of some labels."""

from pydantic import BaseModel, Field

from baseline_core.core.ml.data.taxonomy import Taxonomy


class Prompt(BaseModel):
    """A prompt ready to send, with the labels it asks for.

    Attributes:
        text: What the model is asked.
        labels: Labels the completion may name; others are dropped.
    """

    text: str
    labels: list[str] = Field(min_length=1)


class VlmPromptBuilder:
    """Writes prompts listing the labels to find, with their descriptions, and the line format.

    Attributes:
        taxonomy: Descriptions of every label a prompt may ask for.
    """

    def __init__(self, taxonomy: Taxonomy) -> None:
        """Keep the taxonomy.

        Args:
            taxonomy: Descriptions of every label a prompt may ask for.
        """
        self.taxonomy = taxonomy

    def build(self, labels: list[str], context: str) -> Prompt:
        """The prompt asking for ``labels``: context, descriptions, rules and line format.

        Args:
            labels: Labels asked for, in the taxonomy.
            context: Opening sentence situating the image, e.g. what a crop shows.

        Returns:
            The prompt; a label missing from the taxonomy is a ``ValueError``.
        """
        descriptions = self.taxonomy.subset(labels).descriptions
        label_lines = "\n".join(f"- {name}: {text}" for name, text in descriptions.items())
        text = (
            f"{context} Possible classes (most images contain only a few of them; list a class "
            "ONLY if such an object is really visible, and list it once per object):\n"
            f"{label_lines}\n"
            "Rules: every physical object gets its own line, even when objects touch, overlap "
            "or are stacked (two stacked boxes = two lines, three bags side by side = three "
            "lines). Boxes are amodal: if something covers part of an object, the box still "
            "covers the whole object. Answer with one line per object and nothing else, in this "
            "exact format:\n"
            "label x1 y1 x2 y2\n"
            "where label is one of the class names above and x1 y1 x2 y2 are integers from 0 "
            "to 1000 (top-left and bottom-right corners). Example line:\n"
            "box 250 330 820 730\n"
            "If there is nothing, answer: none"
        )
        return Prompt(text=text, labels=labels)
