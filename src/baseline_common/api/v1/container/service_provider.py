"""Service-layer Dishka provider."""

from dishka import Provider, Scope, from_context

from baseline_common.api.settings import Settings


class ServiceProvider(Provider):
    """Provide the settings the app is built with; a project adds its services next to them."""

    settings = from_context(provides=Settings, scope=Scope.APP)
