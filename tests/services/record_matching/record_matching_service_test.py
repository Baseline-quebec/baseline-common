import pytest

from baseline_common.models.organization import Organization
from baseline_common.models.record_matching import MatchDecision, MatchRequest, MatchVerdict
from baseline_common.services.record_matching.record_matching_service import RecordMatchingService


@pytest.fixture
def service() -> RecordMatchingService:
    """Return the service with the default legal suffixes.

    Returns:
        The service.
    """
    return RecordMatchingService()


class TestMatch:
    def test__when_one_candidate_has_the_same_name__is_a_match(
        self, service: RecordMatchingService
    ) -> None:
        """One candidate with the same name once normalized is the match."""
        candidates = [Organization(id_="7", name="Acme Ind"), Organization(id_="8", name="Beta")]

        decision = service.match(MatchRequest(name="ACME Ind. Inc. (MTL)", candidates=candidates))

        assert decision == MatchDecision(verdict=MatchVerdict.MATCH, record_ids=["7"])

    def test__when_no_candidate_has_the_same_name__is_a_new_record(
        self, service: RecordMatchingService
    ) -> None:
        """No candidate with the same name makes a new record."""
        decision = service.match(
            MatchRequest(name="Gamma", candidates=[Organization(id_="8", name="Beta")])
        )

        assert decision == MatchDecision(verdict=MatchVerdict.NEW_RECORD, record_ids=[])

    def test__when_two_candidates_have_the_same_name__is_uncertain(
        self, service: RecordMatchingService
    ) -> None:
        """Two candidates with the same name leave it to a person, both named."""
        candidates = [Organization(id_="8", name="Beta"), Organization(id_="9", name="BETA inc.")]

        decision = service.match(MatchRequest(name="Beta ltée", candidates=candidates))

        assert decision == MatchDecision(verdict=MatchVerdict.UNCERTAIN, record_ids=["8", "9"])


class TestNormalize:
    def test__when_a_name_has_case_accents_parentheses_and_suffixes__are_left_out(
        self, service: RecordMatchingService
    ) -> None:
        """Case, accents, parentheses and legal suffixes do not count."""
        normalized = service.normalize("Société Énergie (Québec) Ltée")

        assert normalized == "societe energie"

    def test__when_custom_suffixes_are_given__replace_the_default_ones(self) -> None:
        """The suffixes given replace the defaults; an empty list keeps every word."""
        name = "Beta Inc Group"

        assert RecordMatchingService(["group"]).normalize(name) == "beta inc"
        assert RecordMatchingService([]).normalize(name) == "beta inc group"
