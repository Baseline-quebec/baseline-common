"""LineBuilder — a run of characters set as one line, its runs split on style changes."""

from baseline_core.core.llm.document_extraction.layout.layout import BBox, LayoutLine, LayoutSpan
from baseline_core.core.llm.document_extraction.layout.source.char import Char


class LineBuilder:
    """Builds one line, and the runs it is made of, out of the characters set on it."""

    @classmethod
    def build(cls, chars: list[Char]) -> LayoutLine | None:
        """Build one line: characters grouped into runs on style changes.

        Args:
            chars: The line's characters, in painting order.

        Returns:
            The line, or None when it carries no printable text.
        """
        printable = [char for char in chars if char.ch.strip()]

        if not printable:
            return None

        # A space joins the run it follows; the next printed character is judged against the
        # last printed one, so a face that changes across a space still opens a new run.
        groups: list[list[Char]] = [[chars[0]]]
        anchor = chars[0]

        for char in chars[1:]:
            if char.ch.strip():
                if not (char.style.matches(anchor.style) and char.underlined == anchor.underlined):
                    groups.append([])

                anchor = char

            groups[-1].append(char)

        spans = [span for group in groups if (span := cls._span(group)) is not None]
        text = " ".join("".join(span.text for span in spans).split())
        return LayoutLine(
            text=text,
            bbox=BBox(
                x0=min(g.x0 for g in printable),
                y0=min(g.y0 for g in printable),
                x1=max(g.x1 for g in printable),
                y1=max(g.y1 for g in printable),
            ),
            spans=spans,
            turned=printable[0].style.turned,
        )

    @staticmethod
    def _span(chars: list[Char]) -> LayoutSpan | None:
        """Build one run: every character of it, printed as the file spells it.

        Args:
            chars: The run's characters, in painting order.

        Returns:
            The run, or None when it carries no printable text.
        """
        printable = [char for char in chars if char.ch.strip()]

        if not printable:
            return None

        first = printable[0].style
        flags = (LayoutSpan.BOLD if first.bold else 0) | (LayoutSpan.ITALIC if first.italic else 0)
        return LayoutSpan(
            text="".join(char.ch for char in chars),
            size=first.size,
            font=first.font,
            flags=flags,
            char_flags=LayoutSpan.UNDERLINE if printable[0].underlined else 0,
        )
