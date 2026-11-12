"""Fixtures shared by the core and service suites: the fixture PDFs, a workbook, a span maker."""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the folder of the synthetic fixture PDFs."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def link_kinds_pdf(fixtures_dir: Path) -> bytes:
    """Return a two-page PDF carrying one link of each kind: URI, GoToR, Launch, GoTo."""
    return (fixtures_dir / "link_kinds.pdf").read_bytes()


@pytest.fixture
def scanned_page_pdf(fixtures_dir: Path) -> bytes:
    """Return a PDF whose first page is text and whose second is a lone 8x8 painted image."""
    return (fixtures_dir / "scanned_page.pdf").read_bytes()


@pytest.fixture
def no_outline_bytes(fixtures_dir: Path) -> bytes:
    """Return the bytes of a one-page report with no embedded outline."""
    return (fixtures_dir / "no_outline.pdf").read_bytes()


@pytest.fixture
def with_outline_bytes(fixtures_dir: Path) -> bytes:
    """Return the bytes of a four-page report that carries an embedded outline."""
    return (fixtures_dir / "with_outline.pdf").read_bytes()


@pytest.fixture
def workbook_bytes() -> bytes:
    """Return a workbook: a titled statement sheet, an empty sheet, a hidden one, a 9pt one."""
    openpyxl = pytest.importorskip("openpyxl")
    styles = pytest.importorskip("openpyxl.styles")
    book = openpyxl.Workbook()
    statement = book.active
    statement.title = "Bilan"
    statement["A1"] = "Situation financière"
    statement["A1"].font = styles.Font(size=14, bold=True)

    for column, head in zip("ABC", ["Poste", "20XX", "20XY"], strict=True):
        statement[f"{column}3"] = head
        statement[f"{column}3"].font = styles.Font(bold=True)

    statement["A4"] = "Actifs financiers"
    statement["A4"].font = styles.Font(bold=True)
    statement["A5"], statement["B5"], statement["C5"] = "Encaisse  et\nplacements", 100, 200
    statement["A5"].font = styles.Font(italic=True)
    statement["A6"], statement["B6"], statement["C6"] = "Placements", 50, 60
    statement["A8"], statement["B8"] = "Total", 300
    statement["A8"].font = styles.Font(bold=True)
    statement.column_dimensions["B"].width = 20
    book.create_sheet("Vide")
    hidden = book.create_sheet("NA")
    hidden["A1"] = "Ne pas lire"
    hidden.sheet_state = "hidden"
    notes = book.create_sheet("Notes")
    notes["A1"] = "Nomenclature"
    notes["A1"].font = styles.Font(size=10)

    for row in [2, 3]:
        notes[f"A{row}"] = f"Note {row}"
        notes[f"A{row}"].font = styles.Font(size=9)

    buffer = BytesIO()
    book.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def make_span() -> Callable[..., object]:
    """Return a factory that builds parsed runs for the style predicates."""
    layout = pytest.importorskip("baseline_core.core.llm.document_extraction.layout.layout")

    def build(
        text: str,
        *,
        flags: int = 0,
        font: str = "Helvetica",
        char_flags: int = 0,
        size: float = 11.0,
    ) -> object:
        # flags bit 16 = bold, bit 2 = italic; char_flags bit 2 = underline.
        return layout.LayoutSpan(
            text=text, size=size, font=font, flags=flags, char_flags=char_flags
        )

    return build
