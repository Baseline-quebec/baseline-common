"""Char — one placed character, in top-down points."""

from pydantic import BaseModel

from baseline_core.core.llm.document_extraction.layout.source.style import Style


class Char(BaseModel):
    """One printable character with its geometry and declared style, in top-down points."""

    # Not frozen: ``Advances`` and ``UnderlineMarker`` amend it once the whole page is read.

    ch: str
    """The character itself, as the file encodes it."""

    x0: float
    """Left edge of the character's box."""

    y0: float
    """Top edge of the character's box."""

    x1: float
    """Right edge of the character's box."""

    y1: float
    """Bottom edge of the character's box."""

    baseline: float
    """Y the character is set on, which is what tells one printed line from the next."""

    style: Style
    """The face it is set in, shared with every character of its text object."""

    underlined: bool = False
    """Whether a rule of the page runs through it."""

    advance: float = 0.0
    """How far the pen travels along the run's own direction to set this character."""

    blank: bool = False
    """Whether it stands for a text object the text page dropped: a pen move printing nothing."""

    after_jump: bool = False
    """Whether the library invented a space just before it: the two are not one advance apart."""

    @property
    def size(self) -> float:
        """Effective point size of the face this character is set in."""
        return self.style.size
