import pytest

from baseline_common.service.record_matching.matcher.deps import (
    CandidateRecord,
    RecordMatchingDeps,
    WrittenRecord,
)
from baseline_common.service.record_matching.matcher.matcher import RecordMatcher
from baseline_common.service.record_matching.matcher.output import MatchVerdict


@pytest.fixture
def matcher() -> RecordMatcher:
    """Return a matcher with the default legal suffixes.

    Returns:
        The matcher.
    """
    return RecordMatcher()


class TestDecide:
    async def test__one_candidate_with_the_same_name__is_a_match(
        self, matcher: RecordMatcher
    ) -> None:
        """One candidate with the same name once normalized is the match."""
        deps = RecordMatchingDeps(
            written=WrittenRecord(name="ACME Ind. Inc. (MTL)"),
            candidates=[
                CandidateRecord(id_="7", name="Acme Ind"),
                CandidateRecord(id_="8", name="Beta"),
            ],
        )

        decision = await matcher.decide(deps)

        assert decision.verdict is MatchVerdict.MATCH
        assert decision.record_id == "7"

    async def test__no_candidate_with_the_same_name__is_a_new_record(
        self, matcher: RecordMatcher
    ) -> None:
        """No candidate with the same name makes a new record."""
        deps = RecordMatchingDeps(
            written=WrittenRecord(name="Gamma"), candidates=[CandidateRecord(id_="8", name="Beta")]
        )

        decision = await matcher.decide(deps)

        assert decision.verdict is MatchVerdict.NEW_RECORD

    async def test__two_candidates_with_the_same_name__is_uncertain(
        self, matcher: RecordMatcher
    ) -> None:
        """Two candidates with the same name leave it to a person, both named."""
        deps = RecordMatchingDeps(
            written=WrittenRecord(name="Beta ltée"),
            candidates=[
                CandidateRecord(id_="8", name="Beta"),
                CandidateRecord(id_="9", name="BETA inc."),
            ],
        )

        decision = await matcher.decide(deps)

        assert decision.verdict is MatchVerdict.UNCERTAIN
        assert decision.reason is not None
        assert "8, 9" in decision.reason


class TestNormalize:
    def test__case_accents_parentheses_and_suffixes__are_left_out(
        self, matcher: RecordMatcher
    ) -> None:
        """Case, accents, parentheses and legal suffixes do not count."""
        name = "Société Énergie (Québec) Ltée"

        normalized = matcher.normalize(name)

        assert normalized == "societe energie"

    def test__custom_suffixes__replace_the_default_ones(self) -> None:
        """The suffixes given replace the defaults; an empty list keeps every word."""
        name = "Beta Inc Group"

        assert RecordMatcher(["group"]).normalize(name) == "beta inc"
        assert RecordMatcher([]).normalize(name) == "beta inc group"
