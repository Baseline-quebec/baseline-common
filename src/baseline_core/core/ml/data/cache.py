"""Stores of the results of cached transform steps, computed once and read back on later epochs."""

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import torch


class AbstractTransformCache(ABC):
    """Results of cached transform steps, stored by key."""

    @abstractmethod
    def load(self, key: str) -> Any | None:  # noqa: ANN401
        """Return the value stored under ``key``, or ``None`` if there is none.

        Args:
            key: Cache key.
        """

    @abstractmethod
    def save(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Store ``value`` under ``key``, replacing any previous value.

        Args:
            key: Cache key.
            value: Value to store.
        """


class DiskTransformCache(AbstractTransformCache):
    """One ``torch.save`` file per key in a local folder.

    Attributes:
        cache_dir: Folder holding the files; created on first save.
    """

    SUFFIX: ClassVar[str] = ".pt"

    def __init__(self, cache_dir: Path) -> None:
        """Keep the folder.

        Args:
            cache_dir: Folder holding the files.
        """
        self.cache_dir = cache_dir

    def load(self, key: str) -> Any | None:  # noqa: ANN401
        """Return the value stored under ``key``, or ``None`` if there is none."""
        path = self.path_of(key)

        if not path.is_file():
            return None

        # Values hold annotations and tensor subclasses, and only this process writes the folder.
        return torch.load(path, weights_only=False)

    def save(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Write ``value`` under ``key`` via a temporary file; readers never see half a file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            dir=self.cache_dir, suffix=".tmp", delete=False
        ) as handle:
            torch.save(value, handle)

        Path(handle.name).replace(self.path_of(key))

    def path_of(self, key: str) -> Path:
        """File holding the value of ``key``."""
        return self.cache_dir / f"{key}{self.SUFFIX}"
