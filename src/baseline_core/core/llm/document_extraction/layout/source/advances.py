"""Advances — each character as wide as the pen actually advances."""

from collections import Counter
from itertools import pairwise
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.source.char import Char


class AdvancesConfig(BaseModel):
    """How close two characters must sit to be read as one advance apart."""

    model_config = ConfigDict(frozen=True)

    same_run_drift: Annotated[
        float,
        Field(
            description="How far off its neighbour's line, in points, a character may sit and "
            "still be one advance away from it: nothing, give or take the arithmetic of an angle."
        ),
    ] = 0.05


class Advances:
    """Repairs the reader's boxes: the library draws them round the face, not round the advance."""

    def __init__(self, config: AdvancesConfig | None = None) -> None:
        """Bind the drift tolerance.

        Args:
            config: How close two characters must sit to be one advance apart.
        """
        self._config = config or AdvancesConfig()

    def apply(self, chars: list[Char]) -> list[Char]:
        """Set every character's right edge to where the pen actually advances to.

        Args:
            chars: The page's chars, in the order the file paints them.

        Returns:
            The same chars, each one as wide as its own advance.
        """
        # Two characters of one run are exactly one advance apart: the commonest step IS it.
        same_run_drift = self._config.same_run_drift
        steps: dict[tuple[str, float, str], Counter[float]] = {}

        for previous, char in pairwise(chars):
            # A dropped object already spans the spaces it paints; the step to or from one is
            # the width of a tab stop, never the advance of a glyph.
            if previous.blank or char.blank:
                continue

            face = previous.style
            along = char.x0 - previous.x0
            across = char.baseline - previous.baseline
            step = round(along * face.dx + across * face.dy, 2)
            drift = abs(across * face.dx - along * face.dy)

            if (
                char.after_jump
                or char.style.dx != face.dx
                or char.style.dy != face.dy
                or drift > same_run_drift
                or step <= 0
            ):
                continue

            key = (previous.style.font, previous.style.size, previous.ch)
            steps.setdefault(key, Counter())[step] += 1

        advance = {key: taken.most_common(1)[0][0] for key, taken in steps.items()}
        shrink: dict[tuple[str, float], list[float]] = {}

        for char in chars:
            key = (char.style.font, char.style.size, char.ch)
            taken = advance.get(key)
            width = char.x1 - char.x0

            if taken is not None and width > 0:
                shrink.setdefault(key[:2], []).append(taken / width)

        share = {family: sorted(ratios)[len(ratios) // 2] for family, ratios in shrink.items()}

        for char in chars:
            if char.blank:
                continue

            family = (char.style.font, char.style.size)
            taken = advance.get((*family, char.ch))

            if taken is None:
                ratio = share.get(family)
                taken = (char.x1 - char.x0) * ratio if ratio is not None else char.x1 - char.x0

            char.advance = taken
            char.x1 = char.x0 + taken * char.style.dx

        return chars
