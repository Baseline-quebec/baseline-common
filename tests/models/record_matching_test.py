import pytest
from pydantic import ValidationError

from baseline_common.models.organization import Organization
from baseline_common.models.record_matching import MatchRequest


class TestMatchRequest:
    def test__a_name__is_kept_without_its_surrounding_spaces(self) -> None:
        """Spaces a document leaves around a name are not part of it."""
        request = MatchRequest(name="  Atelier Dupuis  ", candidates=[])

        assert request.name == "Atelier Dupuis"

    @pytest.mark.parametrize("name", ["", "   ", "x" * (MatchRequest.NAME_MAX_LENGTH + 1)])
    def test__a_blank_or_overlong_name__is_refused(self, name: str) -> None:
        """A blank name names nothing; one longer than any name is not one."""
        with pytest.raises(ValidationError):
            MatchRequest(name=name, candidates=[])

    def test__too_many_candidates__are_refused(self) -> None:
        """One request compares a bounded number of candidates."""
        known = Organization(id_="1", name="Atelier Dupuis")

        with pytest.raises(ValidationError, match="too_long"):
            MatchRequest(name="Dupuis", candidates=[known] * (MatchRequest.CANDIDATES_MAX + 1))
