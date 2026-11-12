"""BlockBuilder — a page's characters grouped into lines and blocks, against a moving pen."""

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.assembly.line_builder import LineBuilder
from baseline_core.core.llm.document_extraction.layout.assembly.pen import Break, Pen
from baseline_core.core.llm.document_extraction.layout.layout import TextBlock
from baseline_core.core.llm.document_extraction.layout.source.char import Char


class BlockConfig(BaseModel):
    """The distances, in font-sizes, at which a landing character opens a line or a block."""

    model_config = ConfigDict(frozen=True)

    same_line_drift: Annotated[
        float,
        Field(
            description="How far off the pen's own baseline a character still belongs to the "
            "line being set. Read from the pen rather than a page-wide grid, which keeps a "
            "raised ordinal on its line: « n° » sets its « o » four points up."
        ),
    ] = 0.8
    same_block_drift: Annotated[
        float,
        Field(
            description="Past ``same_line_drift`` and within this, the pen has wrapped to the "
            "next line of the same block; further, it has left the block."
        ),
    ] = 1.5
    indent_step: Annotated[
        float,
        Field(
            description="A wrapped line starting this many POINTS right of the one before is "
            "indented, and an indent opens a paragraph."
        ),
    ] = 0.5
    overstrike_radius: Annotated[
        float,
        Field(
            description="The same character re-struck within this of itself is a faked bold: "
            "one character painted twice, never two."
        ),
    ] = 0.1
    line_gap: Annotated[
        float,
        Field(
            description="A jump this far past the pen ends the line: two columns, or a heading "
            "and the text beside it. Read from the pen, never between two boxes, because the "
            "library collapses the box of the spaces it invents."
        ),
    ] = 0.8
    space_gap: Annotated[
        float,
        Field(
            description="A jump this far past the pen, short of ending the line, is a space "
            "the file never printed."
        ),
    ] = 0.15


class BlockBuilder:
    """Groups a page's characters into lines and blocks, in the order they are painted."""

    SPACED_SCRIPTS_END: ClassVar[int] = 0x0700
    """Code point below which a script sets a space between its words: Latin through Arabic."""

    GENERAL_PUNCTUATION: ClassVar[range] = range(0x2000, 0x20D0)
    """Punctuation every script shares — dashes, quotes, bullets, currency signs."""

    def __init__(self, config: BlockConfig | None = None) -> None:
        """Bind the break distances.

        Args:
            config: The distances at which a character opens a line or a block.
        """
        self._config = config or BlockConfig()

    def build(self, chars: list[Char]) -> list[TextBlock]:
        """Group the page's characters into lines and blocks.

        Args:
            chars: The page's chars, in the order the file paints them.

        Returns:
            The blocks of the page, each holding its lines, in painting order.
        """
        overstrike, space_gap = self._config.overstrike_radius, self._config.space_gap
        blocks: list[list[list[Char]]] = []
        pen: Pen | None = None

        for char in chars:
            if pen is None:
                blocks.append([[char]])
                pen = Pen.opening(char)
                continue

            if pen.overstrikes(char, overstrike):
                continue

            opened = self._break(pen, char)

            if opened is Break.BLOCK:
                blocks.append([[]])
            elif opened is Break.LINE:
                blocks[-1].append([])
            elif pen.spacing(char) > space_gap:
                self._implied_space(blocks[-1][-1], char, pen)

            blocks[-1][-1].append(char)
            pen.advance(char, opens_line=opened is not Break.NONE)

        return [
            TextBlock(lines=lines)
            for block in blocks
            if (lines := [line for run in block if (line := LineBuilder.build(run)) is not None])
        ]

    def _break(self, pen: Pen, char: Char) -> Break:
        """Tell what a character opens when it lands, read against the pen.

        Args:
            pen: Where the previous character advanced to, and the run it was set in.
            char: The character landing.

        Returns:
            Nothing, a line, or a block.
        """
        config = self._config

        if pen.turns(char):
            return Break.BLOCK

        offset = abs(pen.offset(char))

        if offset < config.same_line_drift:
            # A jump either way along the line ends it: forward to a column, or back to a new row.
            return Break.LINE if abs(pen.spacing(char)) > config.line_gap else Break.NONE

        if offset <= config.same_block_drift:
            indented = pen.indent(char) > config.indent_step and not pen.bullet
            return Break.BLOCK if indented else Break.LINE

        return Break.BLOCK

    @classmethod
    def _implied_space(cls, line: list[Char], char: Char, pen: Pen) -> None:
        """Print the space the file left implicit, as a jump the pen never covered.

        Args:
            line: The line being set; the space joins it in place.
            char: The character the jump lands on, whose face the space is set in.
            pen: Where the previous character advanced to.
        """
        if pen.previous == " " or not cls._word_spaced(pen.previous):
            return

        line.append(
            Char(
                ch=" ",
                x0=pen.x,
                y0=char.y0,
                x1=char.x0,
                y1=char.y1,
                baseline=char.baseline,
                style=char.style,
            )
        )

    @classmethod
    def _word_spaced(cls, ch: str) -> bool:
        """Tell whether a character comes from a script that sets a space between its words.

        Args:
            ch: The character.

        Returns:
            True for the scripts below ``SPACED_SCRIPTS_END`` and for shared punctuation.
        """
        code = ord(ch)
        return code < cls.SPACED_SCRIPTS_END or code in cls.GENERAL_PUNCTUATION
