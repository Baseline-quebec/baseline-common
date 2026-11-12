"""ColumnReorder — a page printed in two columns, read left column first, then the right."""

from collections.abc import Sequence
from typing import Annotated, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.layout import (
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    TextBlock,
)


class ColumnConfig(BaseModel):
    """The gutter, shares and counts a page must show before it is read as two columns."""

    model_config = ConfigDict(frozen=True)

    enabled: Annotated[
        bool,
        Field(
            description="Whether the parse reorders two-column pages at all; off, it never looks."
        ),
    ] = True
    gutter_min_width: Annotated[
        float,
        Field(
            description="Narrowest vertical gap, in points, that separates two columns; "
            "most gutters leave 25 to 37 pt, the narrowest seen about 10."
        ),
    ] = 8.0
    min_one_sided_share: Annotated[
        float,
        Field(
            description="Share of the page's lines that must sit wholly on one side of the "
            "gutter; a table whose cells straddle the gap falls far below."
        ),
    ] = 0.75
    min_lines_per_column: Annotated[
        int,
        Field(
            description="Lines each column must carry, so a signature block or a date set right "
            "on an otherwise one-column page is never a column."
        ),
    ] = 6
    centre_band: Annotated[
        tuple[float, float],
        Field(
            description="Where a gutter may sit, as shares of the page width; a gap nearer an "
            "edge is a margin note or a number column, not a gutter."
        ),
    ] = (0.3, 0.7)
    min_balance: Annotated[
        float,
        Field(
            description="Narrower column's width over the wider one's. Two languages set side "
            "by side share the page evenly; a form's label column beside its values does not."
        ),
    ] = 0.6
    gutter_slack: Annotated[
        int,
        Field(
            description="Lines that may sit inside the gutter beyond the full-width ones: a "
            "page number centred in it, a « -3- » set across it."
        ),
    ] = 2
    table_cell_min_width: Annotated[
        float,
        Field(
            description="Narrowest sub-column, in points, that makes a column a table's cells: "
            "wider than a hanging item number, so prose split off one is never a table."
        ),
    ] = 36.0
    table_max_crossing_share: Annotated[
        float,
        Field(
            description="Share of a column's lines that may cross the gap between two of its "
            "cells: a table's caption or a header spanning two cells, nothing like a paragraph."
        ),
    ] = 0.05


class Gutter(NamedTuple):
    """The vertical gap two columns leave between them, in points from the left edge."""

    x0: float
    x1: float


class ColumnReorder:
    """Reads a two-column page column by column, each full-width line keeping its place."""

    LEFT, RIGHT, ROW = 1, 2, 0
    """Where a line falls in a zone: the separator row that opens it, the left or right column."""

    def __init__(self, config: ColumnConfig | None = None) -> None:
        """Bind the detection thresholds.

        Args:
            config: The gutter, shares and counts a two-column page must show.
        """
        self._config = config or ColumnConfig()

    def reorder(self, document: LayoutDocument) -> LayoutDocument:
        """Return the document with every two-column page in reading order.

        Args:
            document: The parsed tree, in the order the file paints it.

        Returns:
            A new tree; the same one when no page changes, or when the reorder is switched off.
        """
        if not self._config.enabled:
            return document

        pages = [self.reorder_page(page) for page in document.pages]

        if all(new is old for new, old in zip(pages, document.pages, strict=True)):
            return document

        return document.model_copy(update={"pages": pages})

    def reorder_page(self, page: LayoutPage) -> LayoutPage:
        """Return one page in reading order: the left column of each zone, then the right.

        Args:
            page: The page, its blocks in painting order.

        Returns:
            A new page when it is two columns painted out of order; the same page otherwise.
        """
        lines = page.lines
        gutter = self.gutter(lines, page.size.width)

        if gutter is None:
            return page

        keys = self._keys(lines, gutter)
        order = sorted(range(len(lines)), key=lambda index: (*keys[index], index))

        if order == list(range(len(lines))):
            return page

        return page.model_copy(update={"blocks": self._regroup(page.blocks, order, keys)})

    def gutter(self, lines: Sequence[LayoutLine], width: float) -> Gutter | None:
        """Find the gutter of a two-column page, or prove the page has none.

        Args:
            lines: Every line of the page.
            width: Page width, in points.

        Returns:
            The gap between the two columns; None when the page does not read as two columns.
        """
        config = self._config

        if len(lines) < 2 * config.min_lines_per_column or width <= 0:
            return None

        gap = self._widest_gap(lines, width)

        if gap is None or gap.x1 - gap.x0 < config.gutter_min_width:
            return None

        left = [line for line in lines if self._side(line, gap) == self.LEFT]
        right = [line for line in lines if self._side(line, gap) == self.RIGHT]

        if min(len(left), len(right)) < config.min_lines_per_column:
            return None

        if (len(left) + len(right)) / len(lines) < config.min_one_sided_share:
            return None

        widths = sorted(self._extent(column) for column in (left, right))
        # Uneven sides are a form; sides each split into cells are a table, whose gap is one
        # of its cell gaps: either way each row must stay whole.
        uneven = widths[0] < config.min_balance * widths[1]

        if uneven or (self._has_cells(left) and self._has_cells(right)):
            return None

        return gap

    def _has_cells(self, column: Sequence[LayoutLine]) -> bool:
        """Tell whether a column is itself split into two sub-columns of table cells.

        Args:
            column: The lines of one column, never empty.

        Returns:
            True when a gap no paragraph crosses parts it into two sub-columns of cells.
        """
        config = self._config
        low = int(min(line.bbox.x0 for line in column))
        high = int(max(line.bbox.x1 for line in column))
        delta = [0] * (high - low + 2)

        for line in column:
            start, end = int(line.bbox.x0) + 1 - low, int(line.bbox.x1) + 1 - low

            if start < end:
                delta[start] += 1
                delta[end] -= 1

        allowed = config.table_max_crossing_share * len(column)
        running, opened = 0, None

        for offset, step in enumerate(delta):
            running += step
            clear = running <= allowed and offset < len(delta) - 1

            if clear and opened is None:
                opened = offset
            elif not clear and opened is not None:
                if offset - opened >= config.gutter_min_width and self._cells_either_side(
                    column, Gutter(float(low + opened), float(low + offset))
                ):
                    return True

                opened = None

        return False

    def _cells_either_side(self, column: Sequence[LayoutLine], gap: Gutter) -> bool:
        """Tell whether a gap inside a column has a sub-column of table cells on each side.

        Args:
            column: The lines of one column.
            gap: A gap inside it that no paragraph crosses.

        Returns:
            True when each side carries enough lines and is wider than a hanging item number.
        """
        config = self._config
        left = [line for line in column if line.bbox.x1 <= gap.x1 and line.bbox.x0 < gap.x0]
        right = [line for line in column if line.bbox.x0 >= gap.x0 and line.bbox.x1 > gap.x1]
        return all(
            len(cells) >= config.min_lines_per_column
            and self._extent(cells) >= config.table_cell_min_width
            for cells in (left, right)
        )

    def _widest_gap(self, lines: Sequence[LayoutLine], width: float) -> Gutter | None:
        """Find the widest run of the centre band crossed by the fewest lines.

        Args:
            lines: Every line of the page.
            width: Page width, in points.

        Returns:
            The widest least-crossed run, in whole points; None when even it is crossed too often.
        """
        low, high = (int(share * width) for share in self._config.centre_band)
        delta = [0] * (high - low + 2)

        for line in lines:
            start, end = max(low, int(line.bbox.x0) + 1), min(high + 1, int(line.bbox.x1) + 1)

            if start < end:
                delta[start - low] += 1
                delta[end - low] -= 1

        crossed, running = [], 0

        for step in delta[:-1]:
            running += step
            crossed.append(running)

        # Only full-width lines cross a gutter, so the floor is their count; the slack admits
        # what sits in it (a centred page number), never a ragged column end.
        floor = min(crossed)

        if floor > len(lines) * (1 - self._config.min_one_sided_share):
            return None

        ceiling = floor + self._config.gutter_slack
        best: Gutter | None = None
        opened: int | None = None

        for offset, count in enumerate([*crossed, ceiling + 1]):
            if count <= ceiling and opened is None:
                opened = offset
            elif count > ceiling and opened is not None:
                best = self._wider(best, Gutter(float(low + opened), float(low + offset)))
                opened = None

        return best

    @staticmethod
    def _extent(column: Sequence[LayoutLine]) -> float:
        """Measure how wide a column is set, from its leftmost line start to its rightmost end.

        Args:
            column: The lines of one column, never empty.

        Returns:
            The column's width, in points.
        """
        return max(line.bbox.x1 for line in column) - min(line.bbox.x0 for line in column)

    @staticmethod
    def _wider(best: Gutter | None, candidate: Gutter) -> Gutter:
        """Keep the wider of two gaps, the first found on a tie.

        Args:
            best: The widest gap so far, if any.
            candidate: The gap just closed.

        Returns:
            Whichever is wider.
        """
        if best is None or candidate.x1 - candidate.x0 > best.x1 - best.x0:
            return candidate

        return best

    def _side(self, line: LayoutLine, gutter: Gutter) -> int:
        """Tell which side of the gutter a line sits on, or that it spans it.

        Args:
            line: The line.
            gutter: The gap between the columns.

        Returns:
            ``LEFT`` or ``RIGHT``; ``ROW`` for a line that spans the gutter or sits inside it.
        """
        # The gap is measured on whole points and tolerates a few crossers, so a line counts as
        # one-sided when it stays clear of the far edge rather than the near one.
        if line.bbox.x1 <= gutter.x1 and line.bbox.x0 < gutter.x0:
            return self.LEFT

        if line.bbox.x0 >= gutter.x0 and line.bbox.x1 > gutter.x1:
            return self.RIGHT

        return self.ROW

    def _keys(self, lines: Sequence[LayoutLine], gutter: Gutter) -> list[tuple[int, int]]:
        """Give each line the zone it falls in and its place in that zone.

        Args:
            lines: Every line of the page, in painting order.
            gutter: The gap between the columns.

        Returns:
            One (zone, ``ROW``/``LEFT``/``RIGHT``) key per line, aligned with ``lines``.
        """
        sides = [self._side(line, gutter) for line in lines]
        separators = sorted(
            (line for line, side in zip(lines, sides, strict=True) if side == self.ROW),
            key=lambda line: line.bbox.y0,
        )
        keys: list[tuple[int, int]] = []

        for line, side in zip(lines, sides, strict=True):
            centre = (line.bbox.y0 + line.bbox.y1) / 2
            row = next(
                (
                    index
                    for index, separator in enumerate(separators, start=1)
                    if separator is line or separator.bbox.y0 <= centre <= separator.bbox.y1
                ),
                None,
            )

            if row is not None:
                keys.append((row, self.ROW))
                continue

            zone = sum(
                (separator.bbox.y0 + separator.bbox.y1) / 2 < centre for separator in separators
            )
            keys.append((zone, side))

        return keys

    @staticmethod
    def _regroup(
        blocks: Sequence[TextBlock], order: Sequence[int], keys: Sequence[tuple[int, int]]
    ) -> list[TextBlock]:
        """Rebuild the blocks in the new order, a block cut wherever the order leaves it.

        Args:
            blocks: The page's blocks, in painting order.
            order: Indices into the page's flat line list, in reading order.
            keys: The (zone, place) of each line of that flat list.

        Returns:
            The blocks in reading order; a block's lines kept in one column of one zone stay one.
        """
        flat = [line for block in blocks for line in block.lines]
        painted = [index for index, block in enumerate(blocks) for _ in block.lines]
        owners = [(block, *key) for block, key in zip(painted, keys, strict=True)]
        grouped: list[list[LayoutLine]] = []
        previous: tuple[int, int, int] | None = None

        for index in order:
            if owners[index] != previous:
                grouped.append([])

            grouped[-1].append(flat[index])
            previous = owners[index]

        return [TextBlock(lines=lines) for lines in grouped]
