"""What one section names: the mentions every stage reads, under the project's own facts."""

import re
import unicodedata
from datetime import date
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field


class MentionRole(StrEnum):
    """Whether a section owns a mention or only refers to it."""

    OWN = "own"
    CITED = "cited"


class Mention(BaseModel):
    """One thing a section names by a printed value: a code, a place, a party, a reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(
        ...,
        description=(
            "What the value names, in the project's vocabulary of kinds (a contract number, a "
            "site, a standard, an organization). Use only the kinds the instructions list."
        ),
    )
    value: str = Field(
        ...,
        description=(
            "The value exactly as the section prints it, verbatim: never completed, "
            "reformatted or inferred from another section."
        ),
    )
    role: MentionRole = Field(
        ...,
        description=(
            "'own' when the section is filed under this value or acts on it (its own number, "
            "the site it is about); 'cited' when it only refers to it (an earlier report, a "
            "prior agreement it amends). In doubt, 'cited'."
        ),
    )

    @property
    def key(self) -> str:
        """The value reduced to what two printings share: no accents, case, spacing or marks."""
        return self.keyed(self.value)

    @staticmethod
    def keyed(text: str) -> str:
        """Reduce any printing as a mention's key is reduced, to search keys in a whole text.

        Args:
            text: A value, or a whole section to search values in.

        Returns:
            Its letters and digits, accents out, case-folded.
        """
        plain = unicodedata.normalize("NFKD", text)
        stripped = "".join(char for char in plain if not unicodedata.combining(char))
        return re.sub(r"[^0-9a-z]", "", stripped.casefold())


class SectionEntities(BaseModel):
    """The base of every extraction schema: a project subclasses it with its own fields."""

    # Downstream stages read only these two fields; a subclass's own fields are model context.

    model_config = ConfigDict(extra="forbid")

    mentions: list[Mention] = Field(
        default_factory=list,
        description=(
            "Every value the section prints that names a thing of a listed kind, each once. "
            "Leave out a value the section does not print."
        ),
    )
    dated: date | None = Field(
        None,
        description=(
            "The date the section states for itself (issued, signed, decided), never a date it "
            "merely cites; null when it states none."
        ),
    )

    def merged_with(self, other: "SectionEntities") -> Self:
        """Fold another reading of the same section into this one, this one authoritative.

        Args:
            other: The additive reading: a vision pass, or a section a repair folds back in.

        Returns:
            A new object, lists united on their keys and scalars taken from this reading first.
        """
        # An unread section holds the bare base: folded with a read one, the project's schema wins.
        schema = type(other) if issubclass(type(other), type(self)) else type(self)
        merged: dict[str, Any] = {}

        for name in schema.model_fields:
            ours, theirs = getattr(self, name, None), getattr(other, name, None)

            if isinstance(ours, list):
                merged[name] = self._union(ours, theirs if isinstance(theirs, list) else [])
            else:
                merged[name] = (
                    theirs if ours is None or (isinstance(ours, str) and not ours) else ours
                )

        return schema.model_validate(merged)  # type: ignore[return-value]

    @staticmethod
    def _union(first: list[Any], second: list[Any]) -> list[Any]:
        """Concatenate two lists of one field, keeping the first printing of every key.

        Args:
            first: The authoritative entries, kept whole and in order.
            second: The additive entries, kept only when their key is new.

        Returns:
            The merged list.
        """
        seen: set[str] = set()
        merged: list[Any] = []

        for entry in [*first, *second]:
            mark = (
                f"{entry.kind}|{entry.key}"
                if isinstance(entry, Mention)
                else entry.model_dump_json()
                if isinstance(entry, BaseModel)
                else repr(entry)
            )

            if mark in seen:
                continue

            seen.add(mark)
            merged.append(entry)

        return merged
