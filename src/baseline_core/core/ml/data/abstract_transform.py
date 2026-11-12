"""One step of a transform pipeline."""

from abc import ABC, abstractmethod
from typing import Any


class AbstractTransform(ABC):
    """One step of a ``BaseDataset``."""

    @abstractmethod
    def __call__(self, data: Any) -> Any:  # noqa: ANN401
        """Transform ``data`` and return the result."""
