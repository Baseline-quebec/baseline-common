"""Fixtures for the Azure ML launcher tests: a mocked ``MLClient`` and the jobs it returns."""

from unittest.mock import MagicMock

import pytest
from azure.ai.ml import MLClient


@pytest.fixture
def client() -> MagicMock:
    """An ``MLClient`` that never talks to Azure."""
    return MagicMock(spec=MLClient)


@pytest.fixture
def azure_job() -> MagicMock:
    """A completed job, as returned by ``client.jobs.get`` and ``create_or_update``."""
    job = MagicMock()
    job.name = "job-123"
    job.status = "Completed"
    job.studio_url = "https://ml.azure.com/runs/job-123"
    return job
