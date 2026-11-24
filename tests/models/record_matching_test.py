import pytest
from pydantic import ValidationError

from baseline_common.models.organization import Organization
from baseline_common.models.record_matching import MatchRequest


class TestMatchRequest:
    def test__when_a_name_has_surrounding_spaces__returns_it_without_them(self) -> None:
        """Spaces a document leaves around a name are not part of it."""
        request = MatchRequest(name="  Atelier Dupuis  ", candidates=[])

        assert request.name == "Atelier Dupuis"

    @pytest.mark.parametrize("name", ["", "   ", "x" * (MatchRequest.NAME_MAX_LENGTH + 1)])
    def test__when_a_name_is_blank_or_overlong__raises_validation_error(self, name: str) -> None:
        """A blank name names nothing; one longer than any name is not one."""
        with pytest.raises(ValidationError):
            MatchRequest(name=name, candidates=[])

    def test__when_too_many_candidates_are_given__raises_validation_error(self) -> None:
        """One request compares a bounded number of candidates."""
        known = Organization(id_="1", name="Atelier Dupuis")

        with pytest.raises(ValidationError, match="too_long"):
            MatchRequest(name="Dupuis", candidates=[known] * (MatchRequest.CANDIDATES_MAX + 1))
