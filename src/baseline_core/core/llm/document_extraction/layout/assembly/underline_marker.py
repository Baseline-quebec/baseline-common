"""UnderlineMarker — the characters a rule of the page runs through."""

import bisect
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.source.char import Char
from baseline_core.core.llm.document_extraction.layout.source.painted_path import PaintedPath


class UnderlineConfig(BaseModel):
    """Where, in font-sizes around a character, a painted mark is its own underline."""

    model_config = ConfigDict(frozen=True)

    highlight_thickness: Annotated[
        float,
        Field(
            description="A mark thicker than this is a highlight or a background fill, never a "
            "rule: no underline is a quarter of the text tall. Read against the character's "
            "size, so a 4pt stroke rules a 24pt heading and backs a 12pt line."
        ),
    ] = 0.25
    strikeout_reach: Annotated[
        float,
        Field(
            description="Share of the face's own ascent within which a rule ABOVE the baseline "
            "strikes the text out rather than underlining it. Higher than that it is a border, "
            "or the signature line a label prints beneath, and underlines the text under it."
        ),
    ] = 0.8
    min_ascent: Annotated[
        float,
        Field(
            description="Ascent the band is raised to when the face declares less, so a rule "
            "across a space or an underscore is still caught. Raising it further starts "
            "catching the border printed above the text."
        ),
    ] = 0.75
    underline_drop: Annotated[
        float,
        Field(
            description="How far below the baseline a mark may sit and still rule the text: "
            "0.55 misses underlines, 0.70 flags text no rule underlines. Read from the baseline, "
            "not the box, so « Report » and « REPORT » answer alike."
        ),
    ] = 0.6


class UnderlineMarker:
    """Recovers underlines from geometry: the library carries no underline bit at all."""

    def __init__(self, config: UnderlineConfig | None = None) -> None:
        """Bind the band thresholds.

        Args:
            config: Where around a character a mark is its own underline.
        """
        self._config = config or UnderlineConfig()

    def mark(self, chars: list[Char], rules: list[PaintedPath]) -> None:
        """Flag the characters a rule of the page runs through.

        Args:
            chars: The page's chars, mutated in place.
            rules: The page's horizontal marks, from ``PaintReader``.
        """
        if not rules:
            return

        rules = sorted(rules, key=lambda one: one.middle)
        middles = [one.middle for one in rules]
        thickness = self._config.highlight_thickness

        for char in chars:
            if not char.ch.strip():
                continue

            top, floor = self._band(char)
            first = bisect.bisect_left(middles, top)
            last = bisect.bisect_right(middles, floor)
            reach = thickness * char.size
            x0, x1 = char.x0, char.x1

            for index in range(first, last):
                rule = rules[index]

                if rule.thickness >= reach or rule.x1 < x0 or rule.x0 > x1:
                    continue

                if self._rules(char, rule, (top, floor)):
                    char.underlined = True
                    break

    def _band(self, char: Char) -> tuple[float, float]:
        """Read the band around a character in which a mark is the character's own rule.

        Args:
            char: The character.

        Returns:
            (top, bottom) in top-down points, measured from the baseline off the face's metrics.
        """
        ascent = char.style.ascender * char.size
        rise = max(0.0, self._config.min_ascent * char.size - ascent)
        return (
            char.baseline - ascent - rise,
            char.baseline + self._config.underline_drop * char.size,
        )

    def _rules(self, char: Char, rule: PaintedPath, band: tuple[float, float]) -> bool:
        """Tell whether one mark underlines one character.

        Args:
            char: The character.
            rule: A horizontal mark of the same page.
            band: The character's stretched box, from ``_band``.

        Returns:
            True when the mark is thin enough, runs through the band, and is not a strikeout.
        """
        if rule.thickness >= self._config.highlight_thickness * char.size:
            return False

        if rule.x1 < char.x0 or rule.x0 > char.x1:
            return False

        middle = rule.middle

        if not band[0] <= middle <= band[1]:
            return False

        above = char.baseline - middle
        return not 0 < above <= self._config.strikeout_reach * char.style.ascender * char.size
