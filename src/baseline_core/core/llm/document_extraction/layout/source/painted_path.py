"""PaintedPath — one painted path, reduced to what a rule needs."""

from pydantic import BaseModel, ConfigDict


class PaintedPath(BaseModel):
    """One painted path: what it covers and how thick its mark is."""

    model_config = ConfigDict(frozen=True)

    x0: float
    """Left edge the path covers, the stroke included, in top-down points."""

    y0: float
    """Top edge."""

    x1: float
    """Right edge."""

    y1: float
    """Bottom edge."""

    thickness: float
    """How thick the painted mark is, the library's stroke inflation taken back off."""

    @property
    def middle(self) -> float:
        """Y the mark is centred on — the line a rule rules along."""
        return (self.y0 + self.y1) / 2
