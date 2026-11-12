"""Collator delegating to a Hugging Face processor."""

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from baseline_core.core.ml.data.abstract_collator import AbstractCollator
from baseline_core.core.ml.data.batch import Batch


class ProcessorCollatorConfig(BaseModel):
    """How ``ProcessorCollator`` feeds samples to an external processor.

    Attributes:
        inputs: Sample key feeding each processor argument, e.g. ``{"images": "image"}``.
        targets_key: Processor output moved to ``Batch.targets``; the others go to ``Batch.x``.
        metadata_keys: Sample keys copied as lists into ``Batch.metadata``.
        processor_kwargs: Keyword arguments of every processor call; asks for PyTorch tensors.
    """

    inputs: dict[str, str]
    targets_key: str = "labels"
    metadata_keys: list[str] = Field(default_factory=list)
    processor_kwargs: dict[str, Any] = Field(default_factory=lambda: {"return_tensors": "pt"})


class ProcessorCollator(AbstractCollator):
    """Delegate collation to an external processor (a Hugging Face image processor, a tokenizer).

    Attributes:
        config: Which sample keys feed the processor and where its outputs go.
        processor: Callable turning lists of raw values into a mapping of tensors.
    """

    def __init__(
        self, config: ProcessorCollatorConfig, processor: Callable[..., Mapping[str, Any]]
    ) -> None:
        """Store the configuration and the processor.

        Args:
            config: Which sample keys feed the processor and where its outputs go.
            processor: Callable turning lists of raw values into a mapping of tensors.
        """
        self.config = config
        self.processor = processor

    def __call__(self, samples: list[dict[str, Any]]) -> Batch[Any]:
        """Run the processor on ``samples`` and split its outputs into inputs and supervision.

        Args:
            samples: Outputs of ``Dataset.__getitem__``.

        Returns:
            A batch whose ``x`` is the processor output minus ``targets_key``.

        Raises:
            KeyError: If a sample lacks a key named in ``inputs`` or ``metadata_keys``.
        """
        needed = set(self.config.inputs.values()) | set(self.config.metadata_keys)

        for index, sample in enumerate(samples):
            missing = sorted(needed - set(sample))

            if missing:
                error_msg = f"Sample {index} lacks keys {missing} required by the collator."
                raise KeyError(error_msg)

        inputs = {
            kwarg: [sample[key] for sample in samples] for kwarg, key in self.config.inputs.items()
        }
        processed = dict(self.processor(**inputs, **self.config.processor_kwargs))
        targets = processed.pop(self.config.targets_key, None)
        metadata = {key: [sample[key] for sample in samples] for key in self.config.metadata_keys}
        return Batch(x=processed, targets=targets, metadata=metadata or None)
