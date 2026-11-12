"""FurnitureMarker — the running header and footer, proved by recurrence across pages."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.layout import LayoutLine, LayoutPage


class EdgeLine(BaseModel):
    """A line at one edge of a page, reduced to what compares across pages."""

    model_config = ConfigDict(frozen=True)

    key: Annotated[str, Field(description="Normalized text, so two pages compare directly.")]
    band: Annotated[int, Field(description="``bbox.y0`` quantized by the page height.")]

    def matches(self, other: "EdgeLine", tolerance: int) -> bool:
        """Tell whether another page carries this same line at this same height.

        Args:
            other: The edge line of another page, taken at the same end of it.
            tolerance: Band slack allowed between the two positions.

        Returns:
            True when the text and the band both agree.
        """
        return self.key == other.key and abs(self.band - other.band) <= tolerance


class PageEdges(BaseModel):
    """The lines at the top and bottom edges of one page."""

    model_config = ConfigDict(frozen=True)

    top: EdgeLine
    bottom: EdgeLine

    def recurring(
        self, neighbours: list["PageEdges"], *, recurrence: float, tolerance: int
    ) -> frozenset[str]:
        """Keep the edge lines whose text recurs at their own band on the neighbouring pages.

        Args:
            neighbours: Edges of the surrounding pages, this page itself excluded.
            recurrence: Share of the neighbours that must repeat a line for it to be furniture.
            tolerance: Band slack allowed when matching a position across pages.

        Returns:
            The text of this page's running header and footer; empty when neither recurs.
        """
        if not neighbours:
            return frozenset()

        floor = max(1.0, recurrence * len(neighbours))
        return frozenset({
            mine.key
            for mine, theirs in (
                (self.top, [edge.top for edge in neighbours]),
                (self.bottom, [edge.bottom for edge in neighbours]),
            )
            if sum(mine.matches(other, tolerance) for other in theirs) >= floor
        })


class HeaderFooterConfig(BaseModel):
    """Injectable thresholds confirming a line is running header/footer rather than content."""

    model_config = ConfigDict(frozen=True)

    window: Annotated[
        int,
        Field(description="Pages examined on each side when confirming a running header/footer."),
    ] = 8
    recurrence: Annotated[
        float,
        Field(
            description="Share of the window that must repeat an edge line at its band: 0.30 "
            "through 1.00 behave alike, lower starts dropping headings, so 0.5 sits mid-plateau."
        ),
    ] = 0.5
    bins: Annotated[
        int, Field(description="Vertical quantization of a page (200 -> 0.5% of height per band).")
    ] = 200
    tolerance: Annotated[
        int, Field(description="Band slack when matching an edge line's position across pages.")
    ] = 1


class FurnitureMarker:
    """Tells every page which of its edge lines the pages around it print too."""

    def __init__(self, config: HeaderFooterConfig | None = None) -> None:
        """Bind the recurrence thresholds.

        Args:
            config: Window, recurrence, tolerance and quantization.
        """
        self._config = config or HeaderFooterConfig()

    def mark(self, pages: list[LayoutPage]) -> list[LayoutPage]:
        """Give every page its running header/footer verdict.

        Args:
            pages: The parsed pages, in printed order.

        Returns:
            The same pages, each carrying its own verdict.
        """
        config = self._config
        edges = [self._edges(page) for page in pages]
        return [
            page.with_header_footer(
                here.recurring(
                    self._neighbours(edges, index, config.window),
                    recurrence=config.recurrence,
                    tolerance=config.tolerance,
                )
                if (here := edges[index]) is not None
                else frozenset()
            )
            for index, page in enumerate(pages)
        ]

    def _edges(self, page: LayoutPage) -> PageEdges | None:
        """Reduce one page to its topmost and its bottommost line.

        Args:
            page: The parsed page.

        Returns:
            Its two edge lines, or None when the page carries no text.
        """
        lines = page.lines

        if not lines:
            return None

        # Ties broken on the text, so two pages sharing a baseline resolve to the same edge.
        ordered = sorted(lines, key=lambda line: (line.bbox.y0, line.text))
        return PageEdges(
            top=self._edge(ordered[0], page.height), bottom=self._edge(ordered[-1], page.height)
        )

    def _edge(self, line: LayoutLine, height: float) -> EdgeLine:
        """Reduce one line to the text and band another page compares against.

        Args:
            line: The line at one edge of its page.
            height: Height of that page, the band is measured against it.

        Returns:
            The comparable edge line.
        """
        return EdgeLine(
            key=line.furniture_key, band=int(line.bbox.y0 / height * self._config.bins)
        )

    @staticmethod
    def _neighbours(edges: list[PageEdges | None], index: int, window: int) -> list[PageEdges]:
        """Collect the edges of the pages around one page, that page itself excluded.

        Args:
            edges: The edges of every page, in page order; None on a page with no text.
            index: 0-based index of the page under test.
            window: Pages examined on each side of it.

        Returns:
            The edges of the surrounding pages that carry text.
        """
        before = edges[max(0, index - window) : index]
        after = edges[index + 1 : index + 1 + window]
        return [edge for edge in (*before, *after) if edge is not None]
