"""The comparable form two printings share, and the searches that read through it."""

import unicodedata

from rapidfuzz import fuzz


class TextSearch:
    """Stateless needle-in-corpus searches, each explicit about its mode — never a mode flag."""

    @staticmethod
    def simplified(text: str) -> str:
        """Strip accents, lowercase and collapse whitespace: two printings compare equal.

        Args:
            text: Raw text.

        Returns:
            Accent-free, lowercased, single-spaced text.
        """
        stripped = "".join(
            c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
        )
        return " ".join(stripped.lower().split())

    @classmethod
    def exact_lines(cls, needle: str, lines: list[str]) -> list[int]:
        """Every line reading exactly as the needle, under the comparable form.

        Args:
            needle: The text looked for.
            lines: The corpus, one printed line each.

        Returns:
            The indexes, in printed order; empty when the needle reduces to nothing.
        """
        wanted = cls.simplified(needle)

        if not wanted:
            return []

        return [index for index, line in enumerate(lines) if cls.simplified(line) == wanted]

    @classmethod
    def fuzzy_window(cls, needle: str, lines: list[str], start: int) -> tuple[float, int]:
        """Score the window of consecutive lines opening on ``start``, sized to the needle.

        Args:
            needle: The text looked for.
            lines: The corpus, one printed line each.
            start: The line the window opens on.

        Returns:
            The window's score against the needle, and the line it stops before.
        """
        wanted = cls.simplified(needle)
        read = ""
        stop = start + 1

        for index in range(start, len(lines)):
            read = f"{read} {lines[index]}" if read else lines[index]
            stop = index + 1

            if len(cls.simplified(read)) >= len(wanted):
                break

        return fuzz.ratio(wanted, cls.simplified(read)), stop

    @classmethod
    def leading(cls, needle: str, text: str) -> int | None:
        """Where the needle ends when it opens the text, advancing character by character.

        Args:
            needle: The text the corpus may open on.
            text: The corpus.

        Returns:
            The index just past the needle, or None when the text does not open on it.
        """
        wanted = cls.simplified(needle)

        if not wanted:
            return None

        # The text renders the needle with its own spacing, so length would find no boundary.
        accumulated = ""

        for index, character in enumerate(text, start=1):
            accumulated += character

            if cls.simplified(accumulated) == wanted:
                return index

            if len(cls.simplified(accumulated)) > len(wanted):
                return None

        return None
