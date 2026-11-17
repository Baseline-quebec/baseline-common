"""The services' Dishka provider: what the entry points resolve the domain from."""

from dishka import Provider, Scope, provide

from baseline_common.services.record_matching.record_matching_service import RecordMatchingService


class ServiceProvider(Provider):
    """Build the services the entry points resolve; a project declares its own next to them."""

    @provide(scope=Scope.APP)
    @staticmethod
    def record_matching() -> RecordMatchingService:
        """Build the record-matching service, once for the app: it holds no state between calls.

        Returns:
            The service, with the default legal suffixes.
        """
        return RecordMatchingService()
