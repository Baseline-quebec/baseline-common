"""Tests for an ``AbstractBackbone`` implementation."""

from baseline_core.core.ml.data.batch import Batch, TensorTargets
from tests.core.ml.dummy_classes import DummyBackbone


class TestAbstractBackbone:
    """Tests for ``AbstractBackbone`` through ``DummyBackbone``."""

    def test__call__should_return_the_embeddings_and_leave_the_batch_unchanged(
        self, backbone: DummyBackbone, batch: Batch[TensorTargets], hidden_dim: int
    ) -> None:
        """One embedding per input group; the inputs stay as they were."""
        inputs = {group: tensor.clone() for group, tensor in batch.x.items()}

        embeddings = backbone(batch)

        assert set(embeddings) == {"user", "flight"}
        assert embeddings["user"].shape[-1] == hidden_dim
        assert all(tensor.equal(inputs[group]) for group, tensor in batch.x.items())
