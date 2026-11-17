from dishka import make_async_container

from baseline_common.assembly.service_provider import ServiceProvider
from baseline_common.services.record_matching.record_matching_service import RecordMatchingService


class TestServiceProvider:
    async def test__the_record_matching_service__is_built_once_for_the_app(self) -> None:
        """The provider builds the service once; every resolution gets the same object."""
        container = make_async_container(ServiceProvider())

        first = await container.get(RecordMatchingService)
        second = await container.get(RecordMatchingService)
        await container.close()

        assert isinstance(first, RecordMatchingService)
        assert second is first
