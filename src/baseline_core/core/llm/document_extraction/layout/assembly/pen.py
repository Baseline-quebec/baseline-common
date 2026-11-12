"""Pen — where the last character advanced to, and what the next one opens."""

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel

from baseline_core.core.llm.document_extraction.layout.source.char import Char


class Break(Enum):
    """What a character opens when it lands: nothing, a line, or a block."""

    NONE = "none"
    LINE = "line"
    BLOCK = "block"


class Pen(BaseModel):
    """Where the last character advanced to, and the run it was set in."""

    BULLETS: ClassVar[str] = "*·•‣◦▪▫■□●○◆◇►▶➢➤✓✔\ufffd"
    """Characters that open a list item; a digit does too."""

    x: float
    """Where the pen stands along the page."""

    y: float
    """Where the pen stands down the page."""

    dx: float
    """Along-x part of the direction the run is written along."""

    dy: float
    """Along-y part of it."""

    origin_x: float
    """Where the last character started."""

    origin_y: float
    """Baseline the last character was set on."""

    start_x: float
    """Where the line being set started."""

    start_y: float
    """Baseline the line being set started on."""

    previous: str
    """The last character set."""

    bullet: bool = False
    """Whether the line being set opened with a list marker, so its text may hang indented."""

    @classmethod
    def opening(cls, char: Char) -> "Pen":
        """Set the pen on the first character of a page.

        Args:
            char: That character.

        Returns:
            The pen, advanced past it and standing at the head of its line.
        """
        pen = cls(
            x=char.x0,
            y=char.baseline,
            dx=char.style.dx,
            dy=char.style.dy,
            origin_x=char.x0,
            origin_y=char.baseline,
            start_x=char.x0,
            start_y=char.baseline,
            previous=char.ch,
        )
        pen.advance(char, opens_line=True)
        return pen

    def overstrikes(self, char: Char, radius: float) -> bool:
        """Tell whether a character is the last one struck again on top of itself.

        Args:
            char: The character landing.
            radius: How far off the last origin, in font-sizes, still counts as on top of it.

        Returns:
            True for a faked bold: one character painted twice, never two.
        """
        reach = radius * (char.size or 1.0)
        return (
            char.ch == self.previous
            and abs(char.x0 - self.origin_x) < reach
            and abs(char.baseline - self.origin_y) < reach
        )

    def spacing(self, char: Char) -> float:
        """Measure how far along the run a character lands past the pen.

        Args:
            char: The character landing.

        Returns:
            The step along the run, in font-sizes; negative when it lands behind the pen.
        """
        along, across = char.x0 - self.x, char.baseline - self.y
        return (along * self.dx + across * self.dy) / (char.size or 1.0)

    def offset(self, char: Char) -> float:
        """Measure how far off the run's baseline a character lands.

        Args:
            char: The character landing.

        Returns:
            The step across the run, in font-sizes; positive further down the page.
        """
        along, across = char.x0 - self.x, char.baseline - self.y
        return (across * self.dx - along * self.dy) / (char.size or 1.0)

    def indent(self, char: Char) -> float:
        """Measure how far past the head of the current line a character lands.

        Args:
            char: The character landing.

        Returns:
            The step along the run from where the line started, in points.
        """
        return (char.x0 - self.start_x) * self.dx + (char.baseline - self.start_y) * self.dy

    def turns(self, char: Char) -> bool:
        """Tell whether a character is written in another direction than the run.

        Args:
            char: The character landing.

        Returns:
            True when its face writes along another vector.
        """
        return (char.style.dx, char.style.dy) != (self.dx, self.dy)

    def advance(self, char: Char, *, opens_line: bool) -> None:
        """Move the pen past a character just set.

        Args:
            char: The character set.
            opens_line: Whether it opened a line, whose head the pen then remembers.
        """
        if opens_line:
            self.start_x, self.start_y = char.x0, char.baseline
            self.bullet = char.ch.isdigit() or char.ch in self.BULLETS

        self.dx, self.dy = char.style.dx, char.style.dy
        self.x = char.x0 + char.advance * self.dx
        self.y = char.baseline + char.advance * self.dy
        self.origin_x, self.origin_y = char.x0, char.baseline
        self.previous = char.ch
