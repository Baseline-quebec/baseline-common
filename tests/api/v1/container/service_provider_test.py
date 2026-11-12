from dishka import make_async_container

from baseline_common.api.settings import Settings
from baseline_common.api.v1.container.service_provider import ServiceProvider


class TestServiceProvider:
    async def test__the_settings__come_from_the_container_context(self) -> None:
        """The provider takes the settings the app was built with, the same object each time."""
        settings = Settings(app_name="probe")
        container = make_async_container(ServiceProvider(), context={Settings: settings})

        first = await container.get(Settings)
        second = await container.get(Settings)
        await container.close()

        assert first is settings
        assert second is settings
