"""Tests for ``ProcessorCollator``."""

from typing import Any, ClassVar

import pytest
import torch

from baseline_core.core.ml.huggingface.processor_collator import (
    ProcessorCollator,
    ProcessorCollatorConfig,
)
from tests.core.ml.huggingface.dummy_classes import StackingProcessor


class TestProcessorCollator:
    """Tests for ``ProcessorCollator``."""

    EXPECTED_BATCH_SIZE: ClassVar[int] = 2

    @pytest.fixture
    def samples(self) -> list[dict[str, Any]]:
        """Two images with a different number of boxes each."""
        return [
            {
                "image": torch.zeros(3, 4, 4),
                "annotations": {"boxes": torch.zeros(2, 4), "class_labels": torch.tensor([0, 1])},
                "image_id": 7,
            },
            {
                "image": torch.ones(3, 4, 4),
                "annotations": {"boxes": torch.zeros(0, 4), "class_labels": torch.tensor([])},
                "image_id": 8,
            },
        ]

    @pytest.fixture
    def processor(self) -> StackingProcessor:
        """Processor stand-in."""
        return StackingProcessor()

    @pytest.fixture
    def collator(self, processor: StackingProcessor) -> ProcessorCollator:
        """Feed images and annotations to the processor, keep image ids as metadata."""
        return ProcessorCollator(
            ProcessorCollatorConfig(
                inputs={"images": "image", "annotations": "annotations"},
                metadata_keys=["image_id"],
            ),
            processor,
        )

    def test__call__should_put_processor_outputs_in_x_and_supervision_in_targets(
        self, collator: ProcessorCollator, samples: list[dict[str, Any]]
    ) -> None:
        """``targets_key`` is moved to ``targets``; the rest of the output is ``x``."""
        batch = collator(samples)

        assert set(batch.x) == {"pixel_values", "pixel_mask"}
        assert batch.x["pixel_values"].shape[0] == self.EXPECTED_BATCH_SIZE
        assert batch.targets == [sample["annotations"] for sample in samples]

    def test__call__should_gather_metadata_keys(
        self, collator: ProcessorCollator, samples: list[dict[str, Any]]
    ) -> None:
        """Metadata keys are copied as one list per key."""
        batch = collator(samples)

        assert batch.metadata == {"image_id": [7, 8]}

    def test__call__should_forward_processor_kwargs(
        self,
        collator: ProcessorCollator,
        processor: StackingProcessor,
        samples: list[dict[str, Any]],
    ) -> None:
        """Extra processor arguments reach every call; the default asks for PyTorch tensors."""
        collator(samples)

        assert processor.last_kwargs == {"return_tensors": "pt"}

    def test__call__when_sample_lacks_key__should_raise_key_error(
        self, collator: ProcessorCollator, samples: list[dict[str, Any]]
    ) -> None:
        """A sample missing an input or metadata key is reported with its index."""
        del samples[1]["image_id"]

        with pytest.raises(KeyError, match=r"Sample 1 lacks keys \['image_id'\]"):
            collator(samples)
