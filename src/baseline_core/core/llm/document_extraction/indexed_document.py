"""An indexed document: what its file says, and the sections it prints with what each names."""

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.layout.layout import LayoutDocument
from baseline_core.core.llm.document_extraction.section import ResolvedSection


class FileFormat(StrEnum):
    """The formats a parser of `core.llm.document_extraction` turns into a layout tree."""

    PDF = "PDF"
    XLSX = "XLSX"

    @property
    def suffixes(self) -> list[str]:
        """The file-name suffixes a file of this format is saved under, lower-cased."""
        # No ``.xls``: the sheet parser opens Office archives only.
        return {FileFormat.PDF: [".pdf"], FileFormat.XLSX: [".xlsx", ".xlsm"]}[self]


class FileMetadata(BaseModel):
    """What the file itself says: how it is named, how it is set, what its bytes carry."""

    model_config = ConfigDict(frozen=True)

    name: str = Field("", title="File name")
    source: str | None = Field(
        None,
        title=(
            "Where the file was collected from — the URL or path a link annotation names it "
            "by, so a section pointing at it can be told it reached the corpus"
        ),
    )
    format: FileFormat = Field(FileFormat.PDF, title="File format")
    checksum: str | None = Field(
        None,
        title=(
            "SHA-256 of the bytes: the only defense against the same document collected "
            "twice under two names"
        ),
    )
    page_count: int | None = Field(None, gt=0, title="Pages the document carries")
    has_embedded_outline: bool = Field(
        default=False,
        title=(
            "The document carries bookmarks, so its plan has a declared source on top of "
            "the printed format"
        ),
    )
    text_is_extractable: bool = Field(
        default=True,
        title=(
            "Every page that paints content also renders text; False means at least one "
            "page is image-only — scanned — and needs OCR before extraction. A blank page "
            "proves nothing and does not flag. Which pages: ``LayoutPage.text_is_extractable``"
        ),
    )

    @classmethod
    def from_layout(cls, document: LayoutDocument, name: str, source: str | None) -> Self:
        """Read the file's own facts off the tree its bytes parsed into.

        Args:
            document: The parsed document, which already carries its own checksum.
            name: File name; empty when the caller does not know it.
            source: The URL or path the file was collected from; None when unknown.

        Returns:
            The metadata, every derivable field filled.
        """
        return cls(
            name=name,
            source=source,
            format=FileFormat.XLSX if document.origin == "cells" else FileFormat.PDF,
            checksum=document.checksum,
            page_count=document.page_count or None,
            has_embedded_outline=bool(document.outline),
            text_is_extractable=not any(
                page.images and not page.text_is_extractable for page in document.pages
            ),
        )


class IndexedDocument(BaseModel):
    """One document of a corpus, indexed: what its file says, and the sections it prints."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., description="What identifies the document across the corpus.")
    file: FileMetadata = Field(..., title="What the file itself says")
    dated: date | None = Field(
        None,
        title=(
            "The date the system already knows for the document (its collection metadata, its "
            "publication); a section that states no date of its own falls back on it"
        ),
    )
    sections: list[ResolvedSection] = Field(
        default_factory=list,
        title="Kept sections, in reconciliation order, each carrying what it names",
    )
    text: str = Field("", title="Every printed line of the document, in its own order")

    @property
    def section_dates(self) -> list[date | None]:
        """Each section's date: its own, else the last one opened above it, else the document's."""
        resolved: list[date | None] = []
        opened = self.dated

        for section in self.sections:
            opened = section.entities.dated or opened
            resolved.append(opened)

        return resolved
