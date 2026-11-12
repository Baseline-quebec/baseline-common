"""The score of a placement: hard units first, then soft ones."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class Score:
    """A hard and a soft count, compared hard first: one hard unit outweighs any soft count."""

    hard: int = 0
    """The weighted violations of hard constraints; 0 when feasible."""
    soft: int = 0
    """The weighted soft terms."""

    def __add__(self, other: "Score") -> "Score":
        """Both counts summed.

        Args:
            other: The score to add.

        Returns:
            The sum.
        """
        return Score(self.hard + other.hard, self.soft + other.soft)
