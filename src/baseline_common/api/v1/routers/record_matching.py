"""Record-matching routes: which known organization a name written in a document is."""

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter

from baseline_common.models.record_matching import MatchDecision, MatchRequest
from baseline_common.services.record_matching.record_matching_service import RecordMatchingService


router = APIRouter(prefix="/record-matching", tags=["record matching"], route_class=DishkaRoute)


@router.post("")
async def match(
    request: MatchRequest, service: FromDishka[RecordMatchingService]
) -> MatchDecision:
    """Decide which candidate a written name is.

    Args:
        request: The name and the candidates.
        service: Decides by comparing the names once normalized.

    Returns:
        A match, a new record, or uncertain with the candidates that have the name.
    """
    return service.match(request)
