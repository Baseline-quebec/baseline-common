import pytest

from baseline_common.framework.repository.in_memory_repository import InMemoryRepository
from baseline_common.models.organization import Organization
from baseline_common.models.record_matching import MatchVerdict
from baseline_common.orchestrators.record_matching_orchestrator import RecordMatchingOrchestrator
from baseline_common.services.record_matching.record_matching_service import RecordMatchingService


@pytest.fixture
def orchestrator() -> RecordMatchingOrchestrator:
    """Return the orchestrator over two organizations held, the way a project wires it.

    Returns:
        The orchestrator.
    """
    held = [
        Organization(id_="org-1", name="Atelier Dupuis inc."),
        Organization(id_="org-2", name="Usinage Lavoie ltée"),
    ]
    organizations = InMemoryRepository(Organization, held, primary_key="id_")

    return RecordMatchingOrchestrator(organizations, RecordMatchingService())


class TestRecordMatchingOrchestrator:
    async def test__a_name_held__is_matched_to_its_organization(
        self, orchestrator: RecordMatchingOrchestrator
    ) -> None:
        """Every organization held is a candidate: the one whose name it is, is the match."""
        decision = await orchestrator.match("ATELIER DUPUIS")

        assert decision.verdict is MatchVerdict.MATCH
        assert decision.record_ids == ["org-1"]

    async def test__a_name_not_held__is_a_new_record(
        self, orchestrator: RecordMatchingOrchestrator
    ) -> None:
        """No organization held has the name: the document names a new one."""
        decision = await orchestrator.match("Soudure Gagnon")

        assert decision.verdict is MatchVerdict.NEW_RECORD
        assert decision.record_ids == []
