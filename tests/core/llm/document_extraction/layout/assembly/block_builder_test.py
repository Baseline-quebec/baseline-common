"""Unit tests for the block builder: where a landing character opens a line or a block."""

from baseline_core.core.llm.document_extraction.layout.assembly.block_builder import BlockBuilder
from baseline_core.core.llm.document_extraction.layout.source.char import Char
from baseline_core.core.llm.document_extraction.layout.source.style import Style


class TestBuild:
    SIZE = 10.0
    STYLE = Style(font="Body", size=SIZE, weight=400, descriptor=0, ascender=0.9)

    @classmethod
    def word(cls, text: str, x: float, y: float) -> list[Char]:
        """Characters of one word set at a position, each five points wide."""
        return [
            Char(
                ch=ch,
                x0=x + 5.0 * index,
                y0=y - 8.0,
                x1=x + 5.0 * (index + 1),
                y1=y + 2.0,
                baseline=y,
                style=cls.STYLE,
                advance=5.0,
            )
            for index, ch in enumerate(text)
        ]

    @classmethod
    def blank(cls, x: float, y: float, width: float = 0.0) -> Char:
        """A space the text page dropped, standing where the file paints it, as wide as it is."""
        return Char(
            ch=" ",
            x0=x,
            y0=y,
            x1=x + width,
            y1=y,
            baseline=y,
            style=cls.STYLE,
            advance=width,
            blank=True,
        )

    @staticmethod
    def texts(chars: list[Char]) -> list[list[str]]:
        """The lines of each block, as text."""
        return [[line.text for line in block.lines] for block in BlockBuilder().build(chars)]

    def test__a_wrapped_line_stays_in_its_block(self) -> None:
        """A line set one leading below the last continues the block."""
        chars = [*self.word("Un", 100.0, 100.0), *self.word("Deux", 100.0, 112.0)]
        assert self.texts(chars) == [["Un", "Deux"]]

    def test__two_leadings_of_nothing_open_a_block(self) -> None:
        """A jump of two and a half sizes with nothing between is a new paragraph."""
        chars = [*self.word("Un", 100.0, 100.0), *self.word("Deux", 100.0, 125.0)]
        assert self.texts(chars) == [["Un"], ["Deux"]]

    def test__a_blank_line_the_file_sets_keeps_the_block(self) -> None:
        """The same jump spent on a blank line is the page breathing, not a new block."""
        chars = [
            *self.word("Un", 100.0, 100.0),
            self.blank(100.0, 112.5),
            *self.word("Deux", 100.0, 125.0),
        ]
        assert self.texts(chars) == [["Un", "Deux"]]

    def test__a_blank_on_the_line_being_set_changes_nothing(self) -> None:
        """A dropped space painted on the current line neither opens nor moves anything."""
        chars = [
            *self.word("Un", 100.0, 100.0),
            self.blank(300.0, 100.0),
            *self.word("Deux", 100.0, 112.0),
        ]
        assert self.texts(chars) == [["Un", "Deux"]]

    def test__a_tab_stop_the_file_paints_keeps_the_label_on_the_number_line(self) -> None:
        """The pen travels a dropped run of spaces, so the label past it joins the line."""
        # "10" ends at 110; the tab stop paints 110-140; the label starts 3 points past it,
        # thirty past the number, which alone is three sizes: a new line without the tab.
        chars = [
            *self.word("10", 100.0, 100.0),
            self.blank(110.0, 100.0, width=30.0),
            *self.word("Label", 143.0, 100.0),
        ]
        assert self.texts(chars) == [["10 Label"]]

    def test__an_indent_opens_a_paragraph(self) -> None:
        """A wrapped line starting right of the one before it is indented: a new block."""
        chars = [*self.word("Un", 100.0, 100.0), *self.word("Deux", 120.0, 112.0)]
        assert self.texts(chars) == [["Un"], ["Deux"]]

    def test__a_list_item_hangs_its_text_indented(self) -> None:
        """Under a line opening with a digit, the indented continuation stays in the block."""
        chars = [*self.word("1 Un", 100.0, 100.0), *self.word("Deux", 120.0, 112.0)]
        assert self.texts(chars) == [["1 Un", "Deux"]]

    def test__a_jump_back_along_the_baseline_opens_a_line(self) -> None:
        """A character landing well behind the pen on the same baseline starts a new line."""
        chars = [*self.word("Deux", 200.0, 100.0), *self.word("Un", 100.0, 100.0)]
        assert self.texts(chars) == [["Deux", "Un"]]
