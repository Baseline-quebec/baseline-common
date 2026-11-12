"""Fixtures for the document suite: a link-bearing PDF and a furniture-marked page."""

import pytest

from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    LayoutLine,
    LayoutPage,
    LayoutSpan,
    Link,
    PageSize,
    RasterImage,
    TextBlock,
    VectorDrawing,
)


@pytest.fixture
def marked_page() -> LayoutPage:
    """Return a page carrying the furniture verdict, one placement to drop and one to keep."""
    header = LayoutLine(
        text="Rapport technique — révision 3",
        bbox=BBox(x0=72.0, y0=30.0, x1=300.0, y1=42.0),
        spans=[LayoutSpan(text="Rapport technique — révision 3", size=11.0, font="Helvetica")],
    )
    body = LayoutLine(
        text="Résultats des essais",
        bbox=BBox(x0=72.0, y0=400.0, x1=300.0, y1=412.0),
        spans=[LayoutSpan(text="Résultats des essais", size=11.0, font="Helvetica")],
    )
    footer = LayoutLine(
        text="page 3",
        bbox=BBox(x0=72.0, y0=810.0, x1=300.0, y1=822.0),
        spans=[LayoutSpan(text="page 3", size=11.0, font="Helvetica")],
    )
    page = LayoutPage(
        number=3,
        size=PageSize(width=612.0, height=842.0),
        blocks=[TextBlock(lines=[header, body, footer])],
        links=[
            Link(target="https://example.org/", bbox=BBox(x0=72, y0=812, x1=200, y1=820)),
            Link(
                target="https://example.org/PA2025-1.pdf", bbox=BBox(x0=72, y0=402, x1=200, y1=410)
            ),
        ],
        images=[
            RasterImage(bbox=BBox(x0=500, y0=20, x1=560, y1=40)),
            VectorDrawing(bbox=BBox(x0=72, y0=200, x1=400, y1=700)),
        ],
    )
    return page.with_header_footer(frozenset({header.furniture_key, footer.furniture_key}))
