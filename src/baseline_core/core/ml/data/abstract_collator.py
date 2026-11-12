"""Collators turning a list of samples into a batch."""

from abc import ABC, abstractmethod
from typing import Any

from baseline_core.core.ml.data.batch import Batch


class AbstractCollator(ABC):
    """Callable handed to a DataLoader as ``collate_fn``: a list of samples in, one batch out."""

    @abstractmethod
    def __call__(self, samples: list[dict[str, Any]]) -> Batch[Any]:
        """Collate ``samples`` into one batch."""
