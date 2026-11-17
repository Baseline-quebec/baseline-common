"""Doubles for the HTTP client tests: scripted servers standing in for Fidelio and Zoho."""

from collections.abc import Callable
from functools import partial

import pytest

from baseline_common.framework.repository.http.client.http_config import HttpConfig
from tests.framework.repository.http.client.scripted_server import ScriptedServer
from tests.framework.repository.http.client.zoho_double import ZohoDouble


@pytest.fixture
def no_wait() -> HttpConfig:
    """Return the default resilience, minus the backoff a test would otherwise sleep through."""
    return HttpConfig(retry_backoff=0.0)


@pytest.fixture
def server() -> Callable[..., ScriptedServer]:
    """Return a factory building a scripted server."""
    return ScriptedServer


@pytest.fixture
def zoho(no_wait: HttpConfig) -> Callable[..., ZohoDouble]:
    """Return a factory building a Zoho double with scripted grants."""
    return partial(ZohoDouble, no_wait)
