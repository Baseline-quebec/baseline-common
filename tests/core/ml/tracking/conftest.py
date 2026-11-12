"""Fixtures giving every tracking test its own MLflow file store."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A throwaway SQLite MLflow store, active for the duration of the test.

    Yields:
        The tracking URI of the store.
    """
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.end_run()
