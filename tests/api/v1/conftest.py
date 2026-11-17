"""Fixtures of the v1 suite: the versioned app, served over a client that runs its lifespan."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from baseline_common.api.v1.api import init_app, init_container


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Yield a client on a fresh versioned app; the context manager runs its lifespan both ways.

    Yields:
        The client.
    """
    with TestClient(init_app(init_container())) as served:
        yield served
