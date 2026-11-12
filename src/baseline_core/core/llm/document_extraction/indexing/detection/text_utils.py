"""The one text rule detection and the reconciler share: does a line carry a real word?"""

import re
from typing import ClassVar


class TextUtils:
    """Stateless text helpers; never parses a section number, since depth is the model's call."""

    WORD: ClassVar[re.Pattern[str]] = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
    """A word worth reading: three letters or more, digits and underscores excluded."""

    @classmethod
    def has_word(cls, text: str) -> bool:
        """Tell whether a line names anything, or is only numbering and punctuation.

        Args:
            text: A detected candidate or a bookmark title, as printed.

        Returns:
            True when the text carries at least one word of three letters or more.
        """
        return cls.WORD.search(text) is not None
