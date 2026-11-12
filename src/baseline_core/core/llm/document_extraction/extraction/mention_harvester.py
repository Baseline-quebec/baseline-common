"""MentionHarvester — every value a project's formats recognise in a section, found by code."""

import re
from operator import attrgetter
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from baseline_core.core.llm.document_extraction.section_entities import Mention, SectionEntities


class MentionFormat(BaseModel):
    """One printed shape and the kind of mention it always means."""

    model_config = ConfigDict(frozen=True)

    kind: str = Field(..., description="The mention kind every value of this shape is.")
    pattern: str = Field(
        ...,
        description=(
            "The shape, as a regular expression without anchors or inline flags, matched "
            "case-insensitively and whole (never the tail of a longer token). A named group "
            "``value`` keeps only its text, so a label may introduce the code: "
            "``notice\\s+(?P<value>\\d{6,10})``."
        ),
    )

    @model_validator(mode="after")
    def _compiles(self) -> Self:
        """Refuse a pattern that does not compile, at construction rather than at harvest.

        Returns:
            The format, unchanged.
        """
        try:
            re.compile(f"(?:{self.pattern})")
        except re.error as error:
            message = f"format of kind {self.kind!r} does not compile: {error}"
            raise ValueError(message) from error

        return self


class Candidate(BaseModel):
    """One value the formats found printed, the kind its shape always has, and where."""

    model_config = ConfigDict(frozen=True)

    value: str = Field(..., description="The value, its whitespace collapsed.")
    kind: str = Field(..., description="The kind of the format that matched it.")
    start: int = Field(..., ge=0, description="Offset of the value in the harvested text.")
    end: int = Field(..., ge=0, description="Offset just past the value.")


class MentionHarvester:
    """Finds the values a project's formats recognise; a model reading is held to them."""

    BOUNDARY_BEFORE: ClassVar[str] = r"(?<![\w./-])"
    BOUNDARY_AFTER: ClassVar[str] = r"(?![\w-])"
    """A value is harvested whole: never the tail of a longer token, never a run in a number."""

    def __init__(self, formats: list[MentionFormat]) -> None:
        """Bind the formats to harvest with.

        Args:
            formats: The project's shapes, in priority order: at one printing, the first wins.
        """
        self._formats = formats
        self._patterns = [
            re.compile(
                f"{self.BOUNDARY_BEFORE}(?:{mention_format.pattern}){self.BOUNDARY_AFTER}",
                re.IGNORECASE,
            )
            for mention_format in formats
        ]
        self._kinds = {mention_format.kind for mention_format in formats}

    def harvest(self, text: str) -> list[Candidate]:
        """Every value the formats find in ``text``, in printed order, each once.

        Args:
            text: The section's heading and text, as the model will be given them.

        Returns:
            The candidates, the first printing of each value kept.
        """
        found: list[Candidate] = []

        for mention_format, pattern in zip(self._formats, self._patterns, strict=True):
            group: str | int = "value" if "value" in pattern.groupindex else 0

            for hit in pattern.finditer(text):
                value = hit.group(group)

                if value:
                    found.append(
                        Candidate(
                            value=" ".join(value.split()),
                            kind=mention_format.kind,
                            start=hit.start(group),
                            end=hit.end(group),
                        )
                    )

        return self._distinct(found)

    def restrict[Out: SectionEntities](self, entities: Out, candidates: list[Candidate]) -> Out:
        """Retype harvested mentions, drop unharvested ones of a formatted kind, keep the rest.

        Args:
            entities: One section's reading.
            candidates: What the harvest found in that section.

        Returns:
            The restricted reading.
        """
        offered = {Mention.keyed(candidate.value): candidate.kind for candidate in candidates}
        kept = []

        for mention in entities.mentions:
            kind = offered.get(mention.key)

            if kind is not None:
                kept.append(mention.model_copy(update={"kind": kind}))
            elif mention.kind not in self._kinds:
                kept.append(mention)

        return entities.model_copy(update={"mentions": kept})

    @classmethod
    def drop_unprinted[Out: SectionEntities](cls, entities: Out, printed: str) -> Out:
        """Drop from a TEXT reading the mentions its section does not print (never a scan's).

        Args:
            entities: The text reading of one section.
            printed: The section's heading and text, as the reading was given them.

        Returns:
            The reading keeping only mentions whose letters and digits occur in ``printed``.
        """
        haystack = Mention.keyed(printed)
        kept = [
            mention for mention in entities.mentions if mention.key and mention.key in haystack
        ]
        return entities.model_copy(update={"mentions": kept})

    @classmethod
    def _distinct(cls, found: list[Candidate]) -> list[Candidate]:
        """The candidates in printed order, one per value: the first format to match decides.

        Args:
            found: Every match of every format, in format order.

        Returns:
            The candidates kept, in printed order.
        """
        seen: set[str] = set()
        kept: list[Candidate] = []

        # Both sorts are stable: at an equal printing the first format still decides.
        longest = sorted(found, key=attrgetter("end"), reverse=True)

        for candidate in sorted(longest, key=attrgetter("start")):
            key = Mention.keyed(candidate.value)

            if not key or key in seen:
                continue

            # « RPT 2026-03107 » holds « 2026-03107 »: the inner digits are not a second code.
            if any(k.start <= candidate.start and candidate.end <= k.end for k in kept):
                continue

            seen.add(key)
            kept.append(candidate)

        return kept
