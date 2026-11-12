"""RecordMatcher: decides by rules which known record a written record is, if any."""

import re
import unicodedata
from typing import ClassVar

from baseline_common.service.record_matching.matcher.deps import RecordMatchingDeps
from baseline_common.service.record_matching.matcher.output import MatchDecision, MatchVerdict


class RecordMatcher:
    """Decides which candidate a written record is by comparing normalized names."""

    LEGAL_SUFFIXES: ClassVar[list[str]] = [
        "inc",
        "ltd",
        "ltee",
        "llc",
        "corp",
        "co",
        "senc",
        "enr",
    ]
    """Words a legal form adds to a name, left out before comparing."""

    def __init__(self, suffixes: list[str] | None = None) -> None:
        """Keep the legal suffixes the comparison leaves out.

        Args:
            suffixes: The suffixes; defaults to ``LEGAL_SUFFIXES``.
        """
        self.suffixes = set(self.LEGAL_SUFFIXES if suffixes is None else suffixes)

    async def decide(self, deps: RecordMatchingDeps) -> MatchDecision:
        """Decide which candidate the written record is.

        Args:
            deps: The written record and the candidates near it.

        Returns:
            A match when one candidate has the same name, a new record when none has, else uncertain.
        """
        written = self.normalize(deps.written.name)
        same = [
            candidate for candidate in deps.candidates if self.normalize(candidate.name) == written
        ]

        if len(same) == 1:
            return MatchDecision(verdict=MatchVerdict.MATCH, record_id=same[0].id_)

        if not same:
            return MatchDecision(verdict=MatchVerdict.NEW_RECORD)

        ids = ", ".join(candidate.id_ for candidate in same)
        reason = f"Several candidates have the same name once normalized: {ids}."

        return MatchDecision(verdict=MatchVerdict.UNCERTAIN, reason=reason)

    def normalize(self, name: str) -> str:
        """A name without case, accents, punctuation, parentheses or legal suffixes.

        Args:
            name: The name as written.

        Returns:
            The words left, separated by one space.
        """
        plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold()
        plain = re.sub(r"\(.*?\)", " ", plain)
        words = re.sub(r"[^a-z0-9]+", " ", plain).split()

        return " ".join(word for word in words if word not in self.suffixes)
