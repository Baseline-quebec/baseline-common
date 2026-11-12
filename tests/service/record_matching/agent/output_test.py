import pytest
from pydantic import ValidationError

from baseline_core.service.record_matching.agent.output import MatchDecision, MatchVerdict


class TestMatchDecision:
    @pytest.mark.parametrize(
        ("verdict", "record_id"), [(MatchVerdict.MATCH, None), (MatchVerdict.NEW_RECORD, "org-1")]
    )
    def test__an_id_out_of_place__is_refused(
        self, verdict: MatchVerdict, record_id: str | None
    ) -> None:
        """A match without an id, or a new record with one."""
        with pytest.raises(ValidationError, match="record_id"):
            MatchDecision(verdict=verdict, record_id=record_id)

    def test__an_uncertain_verdict__needs_its_reason(self) -> None:
        """A person decides: they need to know what the agent hesitated between."""
        with pytest.raises(ValidationError, match="reason"):
            MatchDecision(verdict=MatchVerdict.UNCERTAIN)

    def test__a_clear_match__says_nothing_more(self) -> None:
        """A leaning belongs to an uncertain verdict only."""
        with pytest.raises(ValidationError, match="uncertain"):
            MatchDecision(verdict=MatchVerdict.MATCH, record_id="org-1", leaning_id="org-2")

    def test__an_uncertain_verdict__may_lean_toward_one_candidate(self) -> None:
        """Uncertain between two, leaning toward one."""
        decision = MatchDecision(
            verdict=MatchVerdict.UNCERTAIN, reason="Same address, two names.", leaning_id="org-1"
        )

        assert decision.leaning_id == "org-1"
