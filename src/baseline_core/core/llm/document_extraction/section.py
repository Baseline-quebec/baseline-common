"""A section of a document's plan: what the reconciler kept, where it was found, what it names."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from baseline_core.core.llm.document_extraction.layout.layout import BBox, Link
from baseline_core.core.llm.document_extraction.section_entities import SectionEntities
from baseline_core.core.llm.document_extraction.text_search import TextSearch


class SectionRelevance(StrEnum):
    """Whether a section concerns what the project extracts, as its criterion words it."""

    RELEVANT = "relevant"
    UNCERTAIN = "uncertain"
    IRRELEVANT = "irrelevant"


class FileReference(BaseModel):
    """A file the section points at, read off the link annotations, never off the text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(
        description="The annotation's destination, as is — a URL, or a file path for a "
        "GoToR or Launch link."
    )
    filename: str = Field(description="The path's last segment (« annex-B.pdf »).")

    @classmethod
    def from_links(cls, links: list[Link], suffixes: list[str]) -> list[Self]:
        """Read the documents a run of link annotations points at, each once.

        Args:
            links: The annotations printed under one heading, in the page's own order.
            suffixes: The lower-cased file-name suffixes a target must end on to be a document.

        Returns:
            One reference per distinct document target; web pages and other formats left out.
        """
        kept: dict[str, Self] = {}

        for link in links:
            filename = link.target.rstrip("/").rsplit("/", 1)[-1]

            if filename.lower().endswith(tuple(suffixes)):
                kept.setdefault(link.target, cls(target=link.target, filename=filename))

        return list(kept.values())


class Section(BaseModel):
    """A section the reconciler kept: its heading split in two, its page, depth and relevance."""

    title: str = Field(
        ...,
        description=(
            "Heading description only, verbatim; the leading ordering enumerator and the "
            "reference code the line OPENS on removed — those go to ``section_number``. A code "
            "the line CLOSES on is part of the description and STAYS."
        ),
    )
    start_page: int = Field(..., title="1-based PHYSICAL start page")
    level: int = Field(
        ...,
        ge=1,
        description=(
            "Depth in the document hierarchy, 1 = top. Decided AFTER the title, from the "
            "depth of the ordering enumerator when a parent is visible. Where nothing is "
            "numbered, 1 is NOT the default: name the nearest heading above that announces "
            "this one and go one level below it. Only a heading nothing above announces is 1."
        ),
    )
    relevance: SectionRelevance = Field(
        ...,
        description=(
            "Whether the section concerns what the instructions' relevance criterion names: "
            "'relevant' on an explicit signal in the heading, 'irrelevant' when the heading "
            "plainly concerns something else, 'uncertain' when the title alone cannot settle "
            "it. In doubt, never 'relevant'."
        ),
    )
    section_number: str | None = Field(
        None,
        description=(
            "Whatever the printed heading OPENS on before its description, verbatim: a bare "
            "ordering enumerator ('1.', '4.', '4.1.', '8.7', 'A.', 'PART I') just as much as "
            "a code anchoring an item ('REF 2026-1397-00', 'Item # : 1234281021'), with the "
            "keyword announcing it, and both in printed order when it opens on both. Reading "
            "the enumerator to decide the level does not consume it. It exists only to locate "
            "where the section STARTS — not an identity, never a level. Null when the heading "
            "opens straight on its description — a code the line CLOSES on is description, it "
            "stays in the title, and it never moves here."
        ),
    )

    @property
    def printed_heading(self) -> str:
        """The heading as the page prints it: the opener the title was split from, restored."""
        if not self.section_number:
            return self.title

        # Both parts may record the code the heading opens on; the page prints it once, so the
        # restored form must too — doubled, it anchors the section on the wrong lines.
        opener, title = self.section_number.split(), self.title.split()
        keys = [TextSearch.simplified(word) for word in title]
        shared = max(
            (
                count
                for count in range(1, min(len(opener), len(title)) + 1)
                if [TextSearch.simplified(word) for word in opener[-count:]] == keys[:count]
            ),
            default=0,
        )
        return " ".join(opener + title[shared:])


class LocatedSection(Section):
    """A kept section, carrying where its heading is printed and the text printed under it."""

    bbox: BBox | None = Field(
        None, title="Extent of the heading line; None when the document carries no such page"
    )
    text: str = Field(
        "", title="Text from the heading down to the next section; empty when nothing was found"
    )
    matched: bool = Field(
        default=False,
        title=(
            "Whether the title was really found on that page; False when the position is only "
            "the best guess the page offered, so the text may open above or below the heading"
        ),
    )
    opener_anchored: bool = Field(
        default=False,
        title=(
            "Whether the anchor is the opener printed alone on its own line — a form's field "
            "bar, so the section opens its page and what prints above the bar is form header"
        ),
    )
    file_references: list[FileReference] = Field(
        default_factory=list,
        title=(
            "The documents the links printed under the heading point at, down to the next "
            "section — harvested off the link annotations, never read from the text"
        ),
    )

    @property
    def body(self) -> str:
        """The text under the title, without the printed heading; empty for a bare rubric."""
        text = self.text.strip()

        for part in [self.section_number or "", self.title]:
            boundary = TextSearch.leading(part, text)

            if boundary is not None:
                text = text[boundary:].lstrip(" \n\t-–—:")  # noqa: RUF001

        return text


class ResolvedSection(LocatedSection):
    """A located section plus what it names: one record, never two lists to re-pair."""

    entities: SerializeAsAny[SectionEntities] = Field(
        default_factory=SectionEntities,
        title="What the section names, in the project's extraction schema; empty when unread",
    )
    extraction_error: str | None = Field(
        None,
        title=(
            "Why reading this section, or one of its vision calls, failed: the section keeps "
            "its place, its text and what was read, and says so, rather than failing its document"
        ),
    )
