"""Unit tests for the line builder: a line's runs split where the face changes."""

from baseline_core.core.llm.document_extraction.layout.assembly.line_builder import LineBuilder
from baseline_core.core.llm.document_extraction.layout.source.char import Char
from baseline_core.core.llm.document_extraction.layout.source.style import Style


class TestBuild:
    BOLD = Style(font="Face-Bold", size=10.0, weight=700, descriptor=0, ascender=0.9)
    BODY = Style(font="Face", size=10.0, weight=400, descriptor=0, ascender=0.9)

    @staticmethod
    def chars(text: str, style: Style, x: float) -> list[Char]:
        """Characters of one word in one face, each five points wide."""
        return [
            Char(
                ch=ch,
                x0=x + 5.0 * index,
                y0=92.0,
                x1=x + 5.0 * (index + 1),
                y1=102.0,
                baseline=100.0,
                style=style,
                advance=5.0,
            )
            for index, ch in enumerate(text)
        ]

    def test__a_face_changing_across_a_space_opens_a_run(self) -> None:
        """A bold lead-in and the body after it are two runs, the space staying with the first."""
        line = LineBuilder.build([
            *self.chars("Ab ", self.BOLD, 0.0),
            *self.chars("cd", self.BODY, 15.0),
        ])
        assert line is not None
        assert [(span.text, span.bold) for span in line.spans] == [("Ab ", True), ("cd", False)]
        assert line.text == "Ab cd"

    def test__a_space_alone_prints_no_line(self) -> None:
        """Characters that print nothing make no line."""
        assert LineBuilder.build(self.chars("  ", self.BODY, 0.0)) is None
