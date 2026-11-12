"""MergeCandidateDetector — gathers the clues that two neighbouring sections COULD be one."""

from enum import StrEnum

from baseline_core.core.llm.document_extraction.classification.section_repair.config import (
    SectionRepairConfig,
)
from baseline_core.core.llm.document_extraction.section import ResolvedSection, SectionRelevance


class Evidence(StrEnum):
    """One clue bringing two neighbouring sections together. None is conclusive."""

    NO_KEY_OF_ITS_OWN = "no_key_of_its_own"
    SAME_KEY = "same_key"
    WEAK_SIGNALS_AGREE = "weak_signals_agree"


class MergeCandidateDetector:
    """Gathers, deterministically, the clues that a boundary between two sections is false."""

    def __init__(self, config: SectionRepairConfig) -> None:
        """Bind the project's identity kinds and the weak-signal threshold.

        Args:
            config: Which mention kinds identify an item, and how many weak signals agree.
        """
        self._config = config

    def evidence_for(self, first: ResolvedSection, second: ResolvedSection) -> list[Evidence]:
        """Gather the clues bringing two neighbouring sections together.

        Args:
            first: The section before the boundary — the absorber, already merged or not.
            second: The section after it.

        Returns:
            The clues found; empty when the pair can never merge or nothing brings it together.
        """
        # Only IRRELEVANT closes a boundary: a keyless fragment split off its item reads uncertain.
        if SectionRelevance.IRRELEVANT in {first.relevance, second.relevance}:
            return []

        first_keys, second_keys = self._identity(first), self._identity(second)

        if first_keys & second_keys:
            return [Evidence.SAME_KEY]

        if first_keys and second_keys:
            return []

        # Two keyless sections attach to nothing; « S, W1, W2 » chains through the merged S+W1.
        if first_keys:
            return [Evidence.NO_KEY_OF_ITS_OWN]

        shared = self._weak_signals(first) & self._weak_signals(second)

        if len(shared) >= self._config.min_shared_signals:
            return [Evidence.WEAK_SIGNALS_AGREE]

        return []

    def _identity(self, section: ResolvedSection) -> set[str]:
        """The identity mentions the section owns — a cited code never counts.

        Args:
            section: The resolved section.

        Returns:
            Their kinds and keys, joined, so two kinds never blur into one.
        """
        return {
            f"{mention.kind}|{mention.key}"
            for mention in section.entities.mentions
            if mention.key and self._config.identifies(mention)
        }

    def _weak_signals(self, section: ResolvedSection) -> set[str]:
        """What the section cites, or names without being identified by it.

        Args:
            section: The resolved section.

        Returns:
            Their kinds and keys, joined, so two kinds never blur into one.
        """
        return {
            f"{mention.kind}|{mention.key}"
            for mention in section.entities.mentions
            if mention.key and not self._config.identifies(mention)
        }
