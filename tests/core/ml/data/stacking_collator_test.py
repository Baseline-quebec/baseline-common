"""Tests for the collators."""

from typing import Any, ClassVar

import pytest
import torch

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.data.stacking_collator import StackingCollator, StackingCollatorConfig


class ImageBatch(Batch[dict[str, torch.Tensor]]):
    """Batch subclass to check ``batch_cls`` is honored."""


class TestStackingCollator:
    """Tests for ``StackingCollator``."""

    EXPECTED_STACKED_SHAPE: ClassVar[tuple[int, int]] = (2, 3)
    EXPECTED_PADDED_SHAPE: ClassVar[tuple[int, int]] = (2, 3)
    PADDING_VALUE: ClassVar[float] = -1.0

    @pytest.fixture
    def samples(self) -> list[dict[str, Any]]:
        """Two samples: a tensor, a number, a string and a variable-length sequence."""
        return [
            {"image": torch.zeros(3), "label": 1, "id": "a", "tokens": torch.tensor([1, 2, 3])},
            {"image": torch.ones(3), "label": 0, "id": "b", "tokens": torch.tensor([4])},
        ]

    @pytest.fixture
    def config(self) -> StackingCollatorConfig:
        """Group every key and pad ``tokens``."""
        return StackingCollatorConfig(
            group_mapping={"x": ["image", "tokens"], "targets": ["label"], "metadata": ["id"]},
            keys_to_pad=["tokens"],
            padding_value=self.PADDING_VALUE,
        )

    @pytest.fixture
    def collator(self, config: StackingCollatorConfig) -> StackingCollator:
        """Collator following ``config``."""
        return StackingCollator(config)

    @pytest.fixture
    def sparse_collator(self) -> StackingCollator:
        """Collator mapping keys the samples do not have."""
        return StackingCollator(
            StackingCollatorConfig(
                group_mapping={"x": ["image", "absent"], "metadata": ["absent"]}
            )
        )

    @pytest.fixture
    def image_batch_collator(self, config: StackingCollatorConfig) -> StackingCollator:
        """Collator building ``ImageBatch`` instances."""
        return StackingCollator(config, batch_factory=ImageBatch)

    def test__call__should_group_keys_per_mapping(
        self, collator: StackingCollator, samples: list[dict[str, Any]]
    ) -> None:
        """Keys land in the batch field ``group_mapping`` assigns them to."""
        batch = collator(samples)

        assert set(batch.x) == {"image", "tokens"}
        assert batch.targets is not None
        assert set(batch.targets) == {"label"}
        assert batch.metadata == {"id": ["a", "b"]}

    def test__collate_field__when_tensors__should_stack(
        self, collator: StackingCollator, samples: list[dict[str, Any]]
    ) -> None:
        """Same-shape tensors are stacked along a new batch dimension."""
        batch = collator(samples)

        assert batch.x["image"].shape == self.EXPECTED_STACKED_SHAPE

    def test__collate_field__when_numbers__should_become_tensor(
        self, collator: StackingCollator, samples: list[dict[str, Any]]
    ) -> None:
        """Python numbers become a tensor."""
        batch = collator(samples)

        assert batch.targets is not None
        assert batch.targets["label"].tolist() == [1, 0]

    def test__collate_field__when_key_to_pad__should_pad_to_longest(
        self, collator: StackingCollator, samples: list[dict[str, Any]]
    ) -> None:
        """Variable-length tensors are padded with ``padding_value``."""
        batch = collator(samples)

        tokens = batch.x["tokens"]
        assert tokens.shape == self.EXPECTED_PADDED_SHAPE
        assert tokens[1].tolist() == [4, self.PADDING_VALUE, self.PADDING_VALUE]

    def test__call__when_mapped_key_absent__should_skip_it(
        self, sparse_collator: StackingCollator, samples: list[dict[str, Any]]
    ) -> None:
        """Keys in the mapping but not in the samples are ignored."""
        batch = sparse_collator(samples)

        assert set(batch.x) == {"image"}
        assert batch.metadata is None

    def test__call__when_custom_batch_factory__should_use_it(
        self, image_batch_collator: StackingCollator, samples: list[dict[str, Any]]
    ) -> None:
        """The configured batch factory builds the batch."""
        batch = image_batch_collator(samples)

        assert isinstance(batch, ImageBatch)
