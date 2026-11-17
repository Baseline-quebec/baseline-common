"""RecordMatchingOrchestrator — which organization held a name written in a document is, if any."""

from baseline_common.framework.repository.abstract_repository import AbstractReadRepository
from baseline_common.models.organization import Organization
from baseline_common.models.record_matching import MatchDecision, MatchRequest
from baseline_common.services.record_matching.record_matching_service import RecordMatchingService


class RecordMatchingOrchestrator:
    """Read the organizations held, then let the service decide which one a written name is."""

    def __init__(
        self, organizations: AbstractReadRepository[Organization], matching: RecordMatchingService
    ) -> None:
        """Bind the organizations held and the service that matches a name against them.

        Args:
            organizations: Holds the known organizations, read whole as the candidates.
            matching: Decides which candidate a name is.
        """
        self.organizations = organizations
        self.matching = matching

    async def match(self, name: str) -> MatchDecision:
        """Decide which organization held a written name is.

        Args:
            name: The name as a document wrote it.

        Returns:
            The service's decision, over every organization held.
        """
        candidates = await self.organizations.get_by_filters(limit=None)

        return self.matching.match(MatchRequest(name=name, candidates=candidates))
