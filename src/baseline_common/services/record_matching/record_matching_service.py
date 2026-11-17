"""RecordMatchingService — which known organization a name written in a document is, if any."""

import re
import unicodedata
from typing import ClassVar

from baseline_common.models.record_matching import MatchDecision, MatchRequest, MatchVerdict


class RecordMatchingService:
    """Decide which candidate a written name is, by comparing names once normalized."""

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

    def match(self, request: MatchRequest) -> MatchDecision:
        """Decide which candidate a written name is.

        Args:
            request: The name as a document wrote it, and the known organizations it may be.

        Returns:
            A match when one candidate has the name, a new record when none has, else uncertain.
        """
        written = self.normalize(request.name)
        ids = [
            candidate.id_
            for candidate in request.candidates
            if self.normalize(candidate.name) == written
        ]

        if not ids:
            return MatchDecision(verdict=MatchVerdict.NEW_RECORD, record_ids=[])

        verdict = MatchVerdict.MATCH if len(ids) == 1 else MatchVerdict.UNCERTAIN

        return MatchDecision(verdict=verdict, record_ids=ids)

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
