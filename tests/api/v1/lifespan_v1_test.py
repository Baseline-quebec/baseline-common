from dishka import AsyncContainer
from fastapi import FastAPI
from pytest_mock import MockerFixture

from baseline_core.api.v1.lifespan_v1 import LifespanV1


class TestLifespanV1:
    async def test__the_container__is_closed_when_the_app_stops(
        self, mocker: MockerFixture
    ) -> None:
        """Closing the container runs its providers' finalizers, once the app has served."""
        container = mocker.AsyncMock(spec=AsyncContainer)
        lifespan = LifespanV1(container)

        async with lifespan.run(FastAPI()):
            container.close.assert_not_awaited()

        container.close.assert_awaited_once()
