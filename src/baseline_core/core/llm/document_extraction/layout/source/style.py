"""Style — the face one text object declares."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class Style(BaseModel):
    """The face one text object declares, read once and shared by every character it paints."""

    model_config = ConfigDict(frozen=True)

    ITALIC_DESCRIPTOR: ClassVar[int] = 1 << 6
    """PDF font descriptor flag set on italic faces."""

    FORCE_BOLD_DESCRIPTOR: ClassVar[int] = 1 << 18
    """PDF font descriptor flag forcing bold rendering at small sizes."""

    BOLD_WEIGHT: ClassVar[int] = 700
    """Font weight at which a face is bold; the CSS convention the descriptor follows."""

    font: str
    """Name of the face, as the file declares it."""

    size: float
    """Effective point size: what the face declares, read through the text matrix."""

    weight: int
    """Declared weight of the face."""

    descriptor: int
    """PDF font descriptor flag bits of the face."""

    ascender: float
    """The face's own ascent, as a share of the size."""

    dx: float = 1.0
    """Along-x part of the unit vector the run is written along, in top-down points."""

    dy: float = 0.0
    """Along-y part of it; zero for upright text, set for a label drawn at an angle."""

    @property
    def turned(self) -> bool:
        """Whether the run is set at an angle rather than across the page."""
        return bool(self.dy)

    @property
    def bold(self) -> bool:
        """Whether the face is bold, by its declared weight or by its ForceBold flag."""
        return self.weight >= self.BOLD_WEIGHT or bool(
            self.descriptor & self.FORCE_BOLD_DESCRIPTOR
        )

    @property
    def italic(self) -> bool:
        """Whether the face is italic or oblique, by its descriptor."""
        return bool(self.descriptor & self.ITALIC_DESCRIPTOR)

    def matches(self, other: "Style") -> bool:
        """Tell whether two runs are set in one face, at one size to a tenth of a point.

        Args:
            other: The style of the run printed beside it.

        Returns:
            True when the two read as a single run of text.
        """
        return (self.font, self.weight, self.descriptor, round(self.size, 1)) == (
            other.font,
            other.weight,
            other.descriptor,
            round(other.size, 1),
        )
