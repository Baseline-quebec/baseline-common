"""SheetHeadingFinder — the headings a workbook states, one sheet judged against its own body."""

from collections import Counter

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    FormatCandidate,
    HeadingCandidates,
    OutlineCandidate,
    StyledBlock,
    Typography,
)
from baseline_core.core.llm.document_extraction.layout.layout import (
    BodyProfile,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
)


class SheetHeadingFinder:
    """Stateless heading detection over a parsed workbook: its sheet names and its set rows."""

    @classmethod
    def scan(cls, document: LayoutDocument) -> HeadingCandidates:
        """Surface both heading sources of one workbook.

        Args:
            document: The workbook, as ``SheetParser`` read it.

        Returns:
            The sheet names as outline entries, and one candidate per stated heading.
        """
        return HeadingCandidates(
            outline=[
                OutlineCandidate(text=entry.title, page=entry.page)
                for entry in document.outline
                if entry.page is not None
            ],
            format=[candidate for page in document.pages for candidate in cls._candidates(page)],
        )

    @classmethod
    def _candidates(cls, page: LayoutPage) -> list[FormatCandidate]:
        """Surface one sheet's headings, in row order.

        Args:
            page: One sheet.

        Returns:
            The rows set above the sheet's own body, alone in its opening column.
        """
        # Per sheet, not per workbook: each sheet may set its body at its own size.
        lines = page.lines
        body = cls._body(lines)
        opening = min((line.bbox.x0 for line in lines), default=0.0)
        found = []

        for line in lines:
            # Alone on its row: a bold total or column head is set like a heading, but not alone.
            if len(line.spans) != 1 or line.bbox.x0 != opening:
                continue

            span = line.spans[0]

            if span.size <= body.size and not (span.bold and not body.bold):
                continue

            block = StyledBlock(
                text=line.text,
                typography=Typography(size=span.size, bold=span.bold, italic=span.italic),
                page=page.number,
                bbox=line.bbox,
            )
            found.append(FormatCandidate.from_block(block, page.height))

        return found

    @staticmethod
    def _body(lines: list[LayoutLine]) -> BodyProfile:
        """Measure how the sheet sets its body.

        Args:
            lines: The sheet's lines.

        Returns:
            The commonest cell size, and whether most cells are bold.
        """
        sizes: Counter[float] = Counter()
        bold = total = 0

        for line in lines:
            for span in line.spans:
                sizes[span.size] += 1
                bold += span.bold
                total += 1

        size = sizes.most_common(1)[0][0] if sizes else 0.0
        return BodyProfile(size=size, bold=bool(total) and bold * 2 > total)
