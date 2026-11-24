from dishka import AsyncContainer
from fastapi import FastAPI
from pytest_mock import MockerFixture

from baseline_common.api.v1.lifespan import lifespan


class TestLifespan:
    async def test__when_the_app_stops__returns_a_closed_container(
        self, mocker: MockerFixture
    ) -> None:
        """Closing the container runs its providers' finalizers, once the app has served."""
        app = FastAPI()
        app.state.dishka_container = mocker.AsyncMock(spec=AsyncContainer)

        async with lifespan(app):
            app.state.dishka_container.close.assert_not_awaited()

        app.state.dishka_container.close.assert_awaited_once()
